//! Parse the host_signal_header buffer the search kernel writes when it
//! finds a candidate matching the PoW target.
//!
//! Byte layout (mirrors `csrc/gemm/host_signal_header.hpp` and Python's
//! `pearl_gemm._ParsedHeader`):
//!
//! ```text
//! offset  size  field
//!   0     4     status (1 = kSignalTriggered)
//!   4    12     gridDim   u32[3]
//!  16    12     blockDim  u32[3]
//!  28    12     blockIdx  u32[3]
//!  40    12     tileCoord u32[3]
//!  52    12     threadIdx u32[3]
//!  64     2     num_registers_per_thread (u16)
//!  66   256     thread_rows u8[256]   (only first N valid)
//! 322   256     thread_cols u8[256]   (only first N valid)
//! 580    12     mma_size      i32[3]
//! 592    12     mma_tile_size i32[3]
//! 604    32     target        u32[8]
//! ```

use std::convert::TryInto;

use crate::error::MinerError;

/// HostSignalStatus enum mirror.
pub const STATUS_IDLE: u32 = 0;
pub const STATUS_TRIGGERED: u32 = 1;

/// Lightweight parser. Only owns slices; doesn't copy thread_rows/cols.
#[derive(Debug)]
pub struct ParsedSignalHeader {
    pub status: u32,
    pub grid_dim: [u32; 3],
    pub block_dim: [u32; 3],
    pub block_idx: [u32; 3],
    pub tile_coord: [u32; 3],
    pub thread_idx: [u32; 3],
    pub num_registers_per_thread: u16,
    /// First `num_registers_per_thread` valid (tile-local row indices, u8).
    pub thread_rows: Vec<u8>,
    /// First `num_registers_per_thread` valid (tile-local col indices, u8).
    pub thread_cols: Vec<u8>,
    pub mma_size: [i32; 3],
    pub mma_tile_size: [i32; 3],
    /// Target round-trip: kernel echoes the target it was given. Useful for
    /// sanity-checking (must match what we wrote to `pow_target_tensor`).
    pub target: [u32; 8],
}

impl ParsedSignalHeader {
    pub fn parse(buf: &[u8]) -> Result<Self, MinerError> {
        if buf.len() < 636 {
            return Err(MinerError::Rpc {
                method: "header_parse".to_string(),
                msg: format!("buffer too small: {} bytes (need >= 636)", buf.len()),
            });
        }
        let u32at = |off: usize| u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let status = u32at(0);
        let grid_dim = [u32at(4), u32at(8), u32at(12)];
        let block_dim = [u32at(16), u32at(20), u32at(24)];
        let block_idx = [u32at(28), u32at(32), u32at(36)];
        let tile_coord = [u32at(40), u32at(44), u32at(48)];
        let thread_idx = [u32at(52), u32at(56), u32at(60)];
        let num_registers_per_thread = u16::from_le_bytes(buf[64..66].try_into().unwrap());
        let n = num_registers_per_thread as usize;
        // Defensive cap: the field is 256 bytes wide. Header sometimes
        // arrives with n=0 (uninitialized) — fall back to 64 (typical
        // value for the sm_80 kernel) for graceful degradation.
        let n_eff = if n == 0 { 64 } else { n.min(256) };
        let thread_rows = buf[66..66 + n_eff].to_vec();
        let thread_cols = buf[322..322 + n_eff].to_vec();

        let i32at = |off: usize| i32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let mma_size = [i32at(580), i32at(584), i32at(588)];
        let mma_tile_size = [i32at(592), i32at(596), i32at(600)];
        let mut target = [0u32; 8];
        for (i, slot) in target.iter_mut().enumerate() {
            *slot = u32at(604 + i * 4);
        }
        Ok(ParsedSignalHeader {
            status,
            grid_dim,
            block_dim,
            block_idx,
            tile_coord,
            thread_idx,
            num_registers_per_thread,
            thread_rows,
            thread_cols,
            mma_size,
            mma_tile_size,
            target,
        })
    }

    pub fn is_triggered(&self) -> bool {
        self.status == STATUS_TRIGGERED
    }
}

/// Global (A_rows, B_cols) the winning thread covered.
///
/// `tile_coord = (tile_m, tile_n, ?)` gives the tile origin in tile units.
/// Multiplied by `mma_tile_size = (TILE_M, TILE_N, _)` gives byte offsets.
/// Then we add the thread-local `thread_rows[i]` / `thread_cols[i]` to get
/// global indices. Dedup + sort.
pub fn extract_indices(h: &ParsedSignalHeader) -> (Vec<usize>, Vec<usize>) {
    let tile_m = (h.tile_coord[0] as usize) * (h.mma_tile_size[0] as usize);
    let tile_n = (h.tile_coord[1] as usize) * (h.mma_tile_size[1] as usize);

    let mut rows: Vec<usize> = h.thread_rows.iter().map(|&r| tile_m + r as usize).collect();
    rows.sort_unstable();
    rows.dedup();

    let mut cols: Vec<usize> = h.thread_cols.iter().map(|&c| tile_n + c as usize).collect();
    cols.sort_unstable();
    cols.dedup();

    (rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_parse_smoke() {
        // Smoke: build a synthetic 640-byte header with status=1, tileCoord=(2,3,0),
        // thread_rows = [4, 12], thread_cols = [0, 1].
        let mut buf = vec![0u8; 640];
        buf[0..4].copy_from_slice(&1u32.to_le_bytes()); // status
                                                        // tileCoord at offset 40
        buf[40..44].copy_from_slice(&2u32.to_le_bytes());
        buf[44..48].copy_from_slice(&3u32.to_le_bytes());
        // num_registers_per_thread at 64
        buf[64..66].copy_from_slice(&2u16.to_le_bytes());
        // thread_rows at 66
        buf[66] = 4;
        buf[67] = 12;
        // thread_cols at 322
        buf[322] = 0;
        buf[323] = 1;
        // mma_tile_size at 592
        buf[592..596].copy_from_slice(&128i32.to_le_bytes());
        buf[596..600].copy_from_slice(&128i32.to_le_bytes());

        let h = ParsedSignalHeader::parse(&buf).unwrap();
        assert!(h.is_triggered());
        assert_eq!(h.tile_coord, [2, 3, 0]);
        assert_eq!(h.num_registers_per_thread, 2);
        assert_eq!(h.thread_rows, vec![4, 12]);
        assert_eq!(h.thread_cols, vec![0, 1]);

        let (rows, cols) = extract_indices(&h);
        // tile_m = 2 * 128 = 256. rows = sorted({256 + 4, 256 + 12}) = [260, 268].
        assert_eq!(rows, vec![260, 268]);
        // tile_n = 3 * 128 = 384. cols = [384, 385].
        assert_eq!(cols, vec![384, 385]);
    }
}

// SPDX-License-Identifier: MIT
// Same pinned resize algorithm as Open-P2P; see third_party/Open-P2P.NOTICE.
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::pixels::PixelType;
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};

/// Resize tightly packed RGB bytes. Caller owns both buffers for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn p2p_resize_rgb(
    source: *const u8,
    source_len: usize,
    width: u32,
    height: u32,
    destination: *mut u8,
    destination_len: usize,
    out_width: u32,
    out_height: u32,
) -> i32 {
    if source.is_null()
        || destination.is_null()
        || width == 0
        || height == 0
        || out_width == 0
        || out_height == 0
    {
        return 1;
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3));
    let wanted = (out_width as usize)
        .checked_mul(out_height as usize)
        .and_then(|n| n.checked_mul(3));
    if expected != Some(source_len) || wanted != Some(destination_len) {
        return 2;
    }
    let bytes = unsafe { std::slice::from_raw_parts(source, source_len) };
    let input = match ImageRef::new(width, height, bytes, PixelType::U8x3) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    let mut output = Image::new(out_width, out_height, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Hamming));
    if Resizer::new()
        .resize(&input, &mut output, Some(&options))
        .is_err()
    {
        return 4;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(output.buffer().as_ptr(), destination, destination_len);
    }
    0
}

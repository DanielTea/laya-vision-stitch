use std::io::{Read, Write};

fn main() {
    let dims: Vec<u32> = std::env::args()
        .skip(1)
        .map(|x| x.parse().expect("integer dimensions"))
        .collect();
    assert_eq!(dims.len(), 4);
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();
    let mut output = vec![0u8; (dims[2] * dims[3] * 3) as usize];
    let status = unsafe {
        stitch_p2p_resize::p2p_resize_rgb(
            input.as_ptr(),
            input.len(),
            dims[0],
            dims[1],
            output.as_mut_ptr(),
            output.len(),
            dims[2],
            dims[3],
        )
    };
    assert_eq!(status, 0);
    std::io::stdout().write_all(&output).unwrap();
}

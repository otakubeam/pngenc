use std::fs::File;
use std::{collections::HashMap, ops::RangeInclusive};
use bitbit::BitWriter;
use std::io::Write;
use lazy_static::lazy_static;

enum LZ77token {
    Match { match_start: u16, match_len: u16 },
    Literal(u8),
}

struct SlidingWindow<'a> {
    buf: &'a [u8],
    start: usize,
    end: usize,
    match_length: usize,
}

impl<'a> SlidingWindow<'a> {
    fn new(data: &'a [u8], len: usize, match_length: usize) -> SlidingWindow<'a> {
        assert!(len <= data.len(), "Window length exceeds data length");
        assert!(len > match_length * 2, "Window length must be greater than twice the match length");
        SlidingWindow {
            buf: data,
            start: 0,
            end: len,
            match_length,
        }
    }

    fn length(&self) -> usize {
        self.end - self.start
    }

    fn get_search_buffer_and(&self) -> (&[u8], &[u8]) {
        let & SlidingWindow {buf, start, end, match_length} = self;
        let threshold = start + self.length() - match_length;
        (&buf[start..threshold], &buf[threshold..end]) 
    }

    fn get_slice(&self) -> &[u8] {
        let & SlidingWindow {buf, start, end, ..} = self;
        &buf[start..end]
    }

    fn try_slide(&mut self, offset: usize) -> bool {
        self.start += offset;
        self.end += offset;
        self.end < self.buf.len()
    }
}

fn compute_hash(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for byte in data.iter() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

struct LZ77encoder<'a> {
    window: SlidingWindow<'a>,
}

impl<'a> Iterator for LZ77encoder<'a> {
    type Item = LZ77token;

    fn next(&mut self) -> Option<Self::Item> {
        if self.window.end > self.window.buf.len() {
            return None;
        }
        let token = self.get_next_token();
        match token {
            LZ77token::Match { match_len, .. } => {
                self.window.try_slide(match_len as usize);
            }
            LZ77token::Literal(_) => {
                self.window.try_slide(1);
            }
        }
        Some(token)
    }
}

impl<'a> LZ77encoder<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        LZ77encoder {
            window: SlidingWindow::new(input, 1 << 15, 256),
        }
    }

    fn get_next_token(&self) -> LZ77token {
        let (search_buffer, bytes_to_match) = self.window.get_search_buffer_and();
        let mut longest_match = 0usize;
        let mut match_offset = 0usize;

        for i in 0..search_buffer.len() {
            let mut matching_len = 0usize;

            while search_buffer.len() > i + matching_len &&
                          search_buffer[i + matching_len] == bytes_to_match[matching_len] 
            {
                matching_len += 1;
                longest_match = std::cmp::max(longest_match, matching_len);
                match_offset = if longest_match == matching_len { i } else { match_offset };
            }
        }

        if longest_match >= 3 {
            LZ77token::Match {
                match_start: match_offset as u16,
                match_len: longest_match as u16,
            }
        } else {
            LZ77token::Literal(bytes_to_match[0])
        }
    }
}

static mut DEFLATE_LENGTH_LOOKUP_TABLE: [u8; 65536] = [0; 65536];
static mut DEFLATE_LENGTH_CODE_TO_SYMBOL: [u16; 1 << 9] = [0; 1 << 9];
static mut DEFLATE_LENGTH_SYMBOL_TO_CODE: [u16; 1 << 9] = [0; 1 << 9];

const MAX_MATCH_LEN: usize = 285;
const MIN_MATCH_LEN: usize = 3;

static MATCH_LEN_TO_SYMBOL: [u16; MAX_MATCH_LEN + 1] = populate_match_len_to_symbol();

const fn populate_match_len_to_symbol() -> [u16; MAX_MATCH_LEN + 1] {
    let mut table = [0u16; MAX_MATCH_LEN + 1];

    let mut i = MIN_MATCH_LEN;
    while i <= MAX_MATCH_LEN {
        table[i] = match i {
            3..=10 => 257 + (i - MIN_MATCH_LEN) as u16,
            11..=12 => 265,
            13..=14 => 266,
            15..=16 => 267,
            17..=18 => 268,
            19..=22 => 269,
            23..=26 => 270,
            27..=30 => 271,
            31..=34 => 272,
            35..=42 => 273,
            43..=50 => 274,
            51..=58 => 275,
            59..=66 => 276,
            67..=82 => 277,
            83..=98 => 278,
            99..=114 => 279,
            115..=130 => 280,
            131..=162 => 281,
            163..=194 => 282,
            195..=226 => 283,
            227..=257 => 284,
            258..=285 => 285,
            _ => panic!("Invalid match length"),
        };

        i += 1;
    }

    table
}

struct DefalteEncoder {

}

fn write_block<W: Write>(w: &mut BitWriter<W>, tokens: &[LZ77token], bfinal: bool) -> std::io::Result<()> {
    // Write BFINAL and BTYPE
    w.write_bits(bfinal as u32, 1)?; // BFINAL
    w.write_bits(0b01, 2)?; // BTYPE: Fixed Huffman

    // Write each token
    for token in tokens {
        write_token(w, token)?;
    }

    // Write the End of Block (EOB) symbol
    let eob_code = unsafe { DEFLATE_LENGTH_SYMBOL_TO_CODE[256] };
    w.write_bits(eob_code as u32, 7)?;

    Ok(())
}

enum ChunkType {
    IHDR, IDAT, IEND = 0xDEADBEEF,
}

/// Specialized IHDR chunk
#[repr(C)]
struct IHDRChunk {
    signature: [u8; 4],
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    compression_method: u8,
    filter_method: u8,
    interlace_method: u8,
}

impl IHDRChunk {
    pub fn new(width: u32, height: u32) -> Self {
        IHDRChunk {
            signature: *b"IHDR", 
            width: width.to_be(),
            height: height.to_be(),
            bit_depth: 8, // Default: 8-bit
            color_type: 2, // Default: Truecolor (RGB)
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const IHDRChunk as *const u8, 
                std::mem::size_of::<IHDRChunk>()
            )
        }
    }
}

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

fn write_png<W: Write>(w: &mut W, data: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    // Write PNG signature
    w.write_all(&PNG_SIGNATURE)?;

    // Write IHDR chunk
    let ihdr = IHDRChunk::new(width, height);
    w.write_all(ihdr.as_bytes())?;

    // Compress image data using your Deflate encoder
    let mut compressed_data = Vec::new();
    {
        let mut bit_writer = bitbit::BitWriter::new(&mut compressed_data);
        let encoder = LZ77encoder::new(data);
        write_deflate_blocks(&mut bit_writer, encoder)?;
    }

    // Write IDAT chunk
    w.write_all(b"IDAT")?;
    w.write_all(&compressed_data.len().to_ne_bytes())?;
    w.write_all(&compressed_data)?;

    w.write_all(b"IEND")?;

    Ok(())
}

fn write_deflate_blocks<W: Write>(w: &mut BitWriter<W>, lz: LZ77encoder<'_>) -> std::io::Result<()> {
    const BLOCK_SIZE: usize = 1 << 15; // 32 KB block size

    let mut block_bytes_written = 0;
    let mut remaining_data = lz.window.buf.len(); // Total data length
    let mut block_tokens = Vec::<LZ77token>::new();

    for token in lz {
        let tok_len = match &token {
            LZ77token::Literal(_) => 1,
            LZ77token::Match { match_len, .. } => *match_len as usize,
        };

        remaining_data -= tok_len;
        block_bytes_written += tok_len;

        block_tokens.push(token);

        // At around 32 KB (1 << 15) or when all data is written, start a new block
        if block_bytes_written >= BLOCK_SIZE || remaining_data == 0 {
            // Determine if this is the final block
            let bfinal = if remaining_data == 0 { true } else { false };

            write_block(w, &block_tokens, bfinal)?;

            block_bytes_written = 0; // Reset block byte counter
            block_tokens.clear();
        }
    }

    Ok(())
}

fn write_token<W: Write>(w: &mut BitWriter<W>, tok: &LZ77token) -> std::io::Result<()> {
    match *tok {
        // Handle literals
        LZ77token::Literal(l) => {
            if l <= 143 {
                w.write_bits(l as u32, 8)?; // Literals with 8-bit representation
            } else if l > 143 {
                w.write_bits(l as u32, 9)?; // Literals with 9-bit representation
            } else {
                unreachable!("Invalid literal value"); // Catch impossible cases
            }
        }, 
        
        // Handle matching references
        LZ77token::Match { match_start, match_len } => {
            let symbol = MATCH_LEN_TO_SYMBOL[match_len as usize];
            let code = unsafe { DEFLATE_LENGTH_SYMBOL_TO_CODE[symbol as usize] };
            let (base_len, extra_bits) = DEFLATE_LENGTH_TABLE[&symbol];

            let bit_length = if (256..=279).contains(&symbol) {
                7 // Short matches
            } else if (280..=285).contains(&symbol) {
                8 // Long matches
            } else {
                panic!("Unexpected symbol range: {}", symbol);
            };

            // Write the code and extra bits
            w.write_bits(code as u32, bit_length)?;
            w.write_bits((match_len - base_len) as u32, extra_bits as usize)?;
            w.write_bits(DEFLATE_DISTANCE_TO_CODE[match_start as usize] as u32, 5)?;
        },
    }
    Ok(())
}

static DEFLATE_DISTANCE_TO_CODE: [u8; 32768] = {
    let mut table = [0; 32768]; // Maximum distance is 32768

    let mut code = 0;
    while code < DEFLATE_CODE_TO_DISTANCE.len() {
        let (extra_bits, range_start) = DEFLATE_CODE_TO_DISTANCE[code];
        let range_size = 1 << extra_bits; // Calculate range size for the current code
        let range_end = range_start + range_size;

        let mut distance = range_start;
        while distance < range_end && distance <= 32768 {
            table[distance as usize - 1] = code as u8; // Map the distance to the current code
            distance += 1;
        }

        code += 1;
    }

    table
};

//     Extra           Extra                Extra
// Code Bits Dist  Code Bits   Dist     Code Bits Distance
// ---- ---- ----  ---- ----  ------    ---- ---- --------
//   0   0    1     10   4     33-48    20    9   1025-1536
//   1   0    2     11   4     49-64    21    9   1537-2048
//   2   0    3     12   5     65-96    22   10   2049-3072
//   3   0    4     13   5     97-128   23   10   3073-4096
//   4   1   5,6    14   6    129-192   24   11   4097-6144
//   5   1   7,8    15   6    193-256   25   11   6145-8192
//   6   2   9-12   16   7    257-384   26   12  8193-12288
//   7   2  13-16   17   7    385-512   27   12 12289-16384
//   8   3  17-24   18   8    513-768   28   13 16385-24576
//   9   3  25-32   19   8   769-1024   29   13 24577-32768
static DEFLATE_CODE_TO_DISTANCE: [(u8, u16); 30] = [ 
    // Map 5 bit code to (extra_bits, range_start) 
    (0, 1),       
    (0, 2),       
    (0, 3),       
    (0, 4),       
    (1, 5),       
    (1, 7),       
    (2, 9),       
    (2, 13),      
    (3, 17),      
    (3, 25),      
    (4, 33),      
    (4, 49),      
    (5, 65),      
    (5, 97),      
    (6, 129),     
    (6, 193),     
    (7, 257),     
    (7, 385),     
    (8, 513),     
    (8, 769),     
    (9, 1025),    
    (9, 1537),    
    (10, 2049),    
    (10, 3073),    
    (11, 4097),   
    (11, 6145),    
    (12, 8193),    
    (12, 12289),   
    (13, 16385),   
    (13, 24577),   
];

// Length table
lazy_static! {
    pub static ref DEFLATE_LENGTH_TABLE: HashMap<u16, (u16, u8)> = {
        let mut table = HashMap::new();
        let lengths = [
            (257, 3, 0), (258, 4, 0), (259, 5, 0), (260, 6, 0), (261, 7, 0), (262, 8, 0),
            (263, 9, 0), (264, 10, 0), 
            (265, 11, 1), (266, 13, 1), (267, 15, 1), (268, 17, 1),
            (269, 19, 2), (270, 23, 2), (271, 27, 2), (272, 31, 2), 
            (273, 35, 3), (274, 43, 3),
            (275, 51, 3), (276, 59, 3), (277, 67, 4), (278, 83, 4), (279, 99, 4), 
            
            (280, 115, 4), (281, 131, 5), (282, 163, 5), (283, 195, 5), (284, 227, 5), (285, 258, 0),
        ];
        for &(code, base, extra_bits) in lengths.iter() {
            table.insert(code, (base, extra_bits));
        }
        table
    };
}

// Distance table
lazy_static! {
    pub static ref DEFLATE_DISTANCE_TABLE: HashMap<u16, (u16, u8)> = {
        let mut table = HashMap::new();
        let distances = [
            (0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 1), (5, 7, 1), (6, 9, 2), 
            (7, 13, 2), (8, 17, 3), (9, 25, 3), (10, 33, 4), (11, 49, 4), (12, 65, 5), 
            (13, 97, 5), (14, 129, 6), (15, 193, 6), (16, 257, 7), (17, 385, 7), (18, 513, 8), 
            (19, 769, 8), (20, 1025, 9), (21, 1537, 9), (22, 2049, 10), (23, 3073, 10), 
            (24, 4097, 11), (25, 6145, 11), (26, 8193, 12), (27, 12289, 12), (28, 16385, 13), 
            (29, 24577, 13),
        ];
        for &(code, base, extra_bits) in distances.iter() {
            table.insert(code, (base, extra_bits));
        }
        table
    };
}

fn populate_deflate_lookup() {
    fn populate_defalte_table_for(code: u16, symbol: u16, bitlen: u8) {
        let start  = code << code.leading_zeros();
        let end = (code + 1) << code.leading_zeros();
        for i in start..end {
            unsafe {
                DEFLATE_LENGTH_LOOKUP_TABLE[i as usize] = bitlen;
                DEFLATE_LENGTH_CODE_TO_SYMBOL[code as usize] = symbol;
                DEFLATE_LENGTH_SYMBOL_TO_CODE[symbol as usize] = code;
            }
        }
    }

    // Summary Table of Ranges by Bit Lengths
    // =======================================
    // Bit Length |  Symbol Range    |  Codes                        | Purpose
    // ------------------------------------------------------------------------------
    //   7 bits	  |  256–279         |  0000000-0010111              | End-of-block and length codes (short matches).
    //   8 bits	  |  0–143,          |  00110000-10111111 (0–143),   | Literals (0–143),
    //               280–285         |  11000000-11000111 (280–287)  | Length codes (long matches)
    //   9 bits	  |  144–255         |  110010000-111111111          | Literals (144–255).

    fn populate_deflate_table_range(r: RangeInclusive<u16>, base_code: u16, bit_length: u8) {
        for (offset, symbol) in r.enumerate() {
            let code = base_code + offset as u16;
            populate_defalte_table_for(code, symbol, bit_length);
        }
    }

    populate_deflate_table_range(256..=279, 0b0000000, 7);
    populate_deflate_table_range(000..=143, 0b00110000, 8);
    populate_deflate_table_range(280..=285, 0b11000000, 8);
    populate_deflate_table_range(144..=255, 0b110010000, 9);
}


fn main() {
    let input = b"abdsoiejow";
    let encoder = LZ77encoder::new(input);
    let data = b"some bytes";

    let mut pos = 0;
    let mut buffer = File::create("foo.txt").expect("...");

    while pos < data.len() {
        let bytes_written = buffer.write(&data[pos..]);
        pos += bytes_written.expect("...");
    }
}
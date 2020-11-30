use std::mem;
use std::time::*;
use rand::thread_rng;
use rand::Rng;
use minifb::{WindowOptions, Window};
use rayon::prelude::*;

struct WorldRule {
    symbols_needed: Vec<u32>,
    output_symbol: u32,
}

struct World {
    size: u32,
    data: Vec<u32>,
    prev_data: Vec<u32>,
    cell_changed_flags: Vec<bool>,
    neighborhood_changed_flags: Vec<bool>,
    symbol_count: u32,
    symbol_to_color: Vec<(u8, u8, u8)>,
    rules: Vec<WorldRule>,
}

impl World {

    pub fn new(world_size:u32, symbol_count:u32, avg_symbols_per_rule:u32, seed:u64) -> World {
        assert!( (world_size as f32).log(2.0) % 1.0 == 0.0, "World size must be a power of 2.");

        // includes end
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut random = |start:u32, end:u32| -> u32 {
            let r: f32 = rng.gen::<f32>();
            start + (r * (end-start) as f32).round() as u32
        };

        let mut symbol_to_color = Vec::new();
        for _i in 0..symbol_count {
            let r = random(0, 255) as u8;
            let g = random(0, 255) as u8;
            let b = random(0, 255) as u8;
            symbol_to_color.push((r, g, b));
        }

        // (indent means equals)
        //
        // > probabiliy of at least one rule match at a position
        //   > one
        //   > minus
        //   > the probability of none of the rules matching at a position
        //     > the product of the probabilities of each rule not matching
        //       > the probability of a random rule rule not matching (i.e. at least one symbol doesn't match)
        //         > one
        //         > minus
        //         > the probability of all the symbols of the rule *matching*
        //           > the probability of a particular symbol existing in a particular neighborhood
        //             > the probability that a symbol is in at least one of the 9 cells
        //               > one
        //               > minus
        //               > the probability that the symbol is in NONE of the squares
        //                 > the probabiliy that a cell doesn't contain a particular symbol
        //                   > 1 - (1 / number_of_symbols)
        //                 > to the power of
        //                 > nine
        //           > to the power of
        //           > the average number of symbols in a rule
        //       > to the power of
        //       > the number of rules
        let mut rule_count = 1;
        let mut prob_match = 0.0;
        while prob_match < 0.999 {
          rule_count *= 2;
          let a = (1.0 - (1.0 / symbol_count as f32)).powf(9.0);
          let b = 1.0 - (1.0 - a).powf(avg_symbols_per_rule as f32);
          prob_match = 1.0 - b.powf(rule_count as f32);
        }

        // #[cfg(feature="interactive")] {
        //     println!("world_size: {}", world_size);
        //     println!("symbol_count: {}", symbol_count);
        //     println!("avg_symbols_per_rule: {}", avg_symbols_per_rule);
        //     println!("rule_count: {}", rule_count);
        //     println!("prob_match: {}", prob_match);
        // }

        let add_symbol_chance = avg_symbols_per_rule as f32 / symbol_count as f32;
        assert!(add_symbol_chance < 1.0);

        let mut world_rules = Vec::new();
        for _i in 0..rule_count {
            let mut symbols_needed = Vec::new();
            for symbol in 0..symbol_count {
                if (random(0, 1000) as f32) < add_symbol_chance*1000.0 { 
                    symbols_needed.push(symbol);
                }
            }
            if symbols_needed.is_empty() {
                symbols_needed.push(random(0, symbol_count-1));
            }
            let output_symbol = random(0, symbol_count-1);
            world_rules.push(WorldRule { symbols_needed, output_symbol });
        }

        assert!( !world_rules.is_empty() );

        World {
            size: world_size,
            data: vec![0; world_size.pow(2) as usize],
            prev_data: vec![0; world_size.pow(2) as usize],
            cell_changed_flags: vec![true; world_size.pow(2) as usize],
            neighborhood_changed_flags: vec![true; world_size.pow(2) as usize],
            symbol_count,
            symbol_to_color,
            rules: world_rules,
        }
    }

    pub fn _set(&mut self, pos:(u32, u32), value:u32) {
        let (x, y) = pos;
        let i = y * self.size + x;
        self.data[i as usize] = value;
    }

    pub fn step(&mut self) {

        mem::swap(&mut self.data, &mut self.prev_data);

        self.cell_changed_flags.iter_mut().for_each(|v| *v = false);

        let world_size = self.size;
        let rules = &self.rules;

        let cell_changed_flags = &mut self.cell_changed_flags;
        let neighborhood_changed_flags = &mut self.neighborhood_changed_flags;
        
        let prev_data = &self.prev_data;
        let data = &mut self.data;

        data.par_iter_mut()
        .zip(cell_changed_flags.par_iter_mut())
        .zip(neighborhood_changed_flags.par_iter()) // <-- don't need iter_mut here.
        .enumerate()
        .for_each(|(i, ((cell, cell_changed_flag), neighborhood_changed_flag))| {
            if !*neighborhood_changed_flag {
                return;
            }
            let x = i as u32 % world_size;
            let y = (i as u32 - x) / world_size;
            let current_value = prev_data[i]; // remember, `prev_data` is "current" value because we did a mem:swap at the start of `step()`
            let next_value = compute_transition(&prev_data, world_size, (x, y), &rules);
            *cell = next_value;
            *cell_changed_flag = next_value != current_value;
        });

        // now we (in effect) run a "erosion" over the `cell_changed_flag` grid to produce the `neighborhood_changed_flag` grid.
        // more concretely: if a cell and all its neighbors are did not change, then we set the neighborhood_changed flag at that
        // postition to false.
        neighborhood_changed_flags.par_iter_mut().enumerate().for_each(|(i, neighborhood_changed_flag)| {
            let xc = i as u32 % world_size;
            let yc = (i as u32 - xc) / world_size;
            for y in (yc as i32 - 1)..(yc as i32 + 2) {
                for x in (xc as i32 - 1)..(xc as i32 + 2) {
                    let yy = if y < 0 {world_size-1} else if y as u32 == world_size {0} else {y as u32};
                    let xx = if x < 0 {world_size-1} else if x as u32 == world_size {0} else {x as u32};
                    let ii = yy*world_size + xx;
                    let changed = cell_changed_flags[ii as usize];
                    if changed {
                        *neighborhood_changed_flag = true;
                        return;
                    }
                }
            }
            *neighborhood_changed_flag = false;
        });

    }

    pub fn randomize(&mut self) {
        let mut rng = thread_rng();
        for i in 0..self.data.len() {
            let r:f32 = rng.gen();
            self.data[i as usize] = (r * self.symbol_count as f32).floor() as u32;
        }
    }

    pub fn draw_to_buffer(&self, buffer:&mut Vec<u32>) {
        let world_size = (self.data.len() as f32).sqrt();
        let window_size = (buffer.len() as f32).sqrt();
        assert!(world_size.log(2.0) % 1.0 == 0.0 && window_size.log(2.0) % 1.0 == 0.0);
        assert!(window_size >= world_size);

        let cell_size = (window_size / world_size) as usize;
        if cell_size == 1 {
            for i in 0..self.data.len() {
                let v = self.data[i as usize];
                let (r, g, b) = self.symbol_to_color[v as usize];
                buffer[i as usize] = (0 as u32) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
            }
        } else {
            // loop over the "cells":
            for y in 0..world_size as usize {
                for x in 0..world_size as usize {
                    let i = y * (world_size as usize) + x;
                    let v = self.data[i as usize];
                    let (r, g, b) = self.symbol_to_color[v as usize];
                    let rgb_bits = (0 as u32) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
                    // fill in this cell:
                    for wy in (y*cell_size)..((y+1)*cell_size) {
                        for wx in (x*cell_size)..((x+1)*cell_size) {
                            let wi = wy * (window_size as usize) + wx;
                            buffer[wi] = rgb_bits;
                        }
                    }
                }
            }           
        }

    }

    pub fn _draw_to_console(&self) {
        use ansi_term::Colour::RGB;
        use ansi_term::ANSIStrings;

        let mut ansi_characters = Vec::new();
        for y in 0..self.size {
            for x in 0..self.size {
                let i = y*self.size + x;
                let v = self.data[i as usize];
                let (r, g, b) = self.symbol_to_color[v as usize];
                let c = RGB(r, g, b).paint("▓▓");
                ansi_characters.push(c);
            }
            ansi_characters.push(RGB(0, 0, 0).paint("\n"));
        }
        println!("{}", ANSIStrings(&ansi_characters));
    }
}

fn compute_transition(prev_data: &[u32], world_size:u32, pos:(u32, u32), rules: &[WorldRule]) -> u32 {
    use std::collections::HashSet;

    unsafe { scratch_counter_1 += 1; }

    // count symbols in neighborhood:
    let (xc, yc) = pos;
    let mut symbol_counts_set = HashSet::<u32>::with_capacity(9);
    for y in (yc as i32 - 1)..(yc as i32 + 2) {
        for x in (xc as i32 - 1)..(xc as i32 + 2) {
            let yy = if y < 0 {world_size-1} else if y as u32 == world_size {0} else {y as u32};
            let xx = if x < 0 {world_size-1} else if x as u32 == world_size {0} else {x as u32};
            let i = yy*world_size + xx;
            let v = prev_data[i as usize];
            symbol_counts_set.insert(v);
            unsafe { scratch_counter_2 += 1; }
        }
    }

    // find first rule that matches:
    for rule in rules.iter() {
        let mut found_non_match = false;
        for symbol in rule.symbols_needed.iter() {
            unsafe { scratch_counter_3 += 1; }
            if !symbol_counts_set.contains(&symbol) {
                found_non_match = true;
                break;
            }
        }
        if !found_non_match {
            return rule.output_symbol;
        }
    }

    // by default keep the same value:
    let i = yc*world_size + xc;
    prev_data[i as usize]
}

static mut scratch_counter_1: u32 = 0;
static mut scratch_counter_2: u32 = 0;
static mut scratch_counter_3: u32 = 0;

fn main() {

    #[cfg(not(feature="interactive"))] {
        println!("# use `cargo run --features \"interactive\" --release` to visually display the worlds (ESC to go to next world; ENTER to replay current world; S to save all frames so far (up to 1000) into gif; P to pause simulation for one second)");
    }

//    remember, goal is to learn rust!
   
//    next up:
//     - perf analysis on current code (GLOBAL to measure how many times parts of code occur each loop - go single-threaded with taskset)
//     - need to make a REALLY good "novelty search" / reward function somehow. manual curation is your bottle-neck.
//     - would "ranges" add anything? conway seems to get a lot of "emergence" out of them
//       OH BECAUSE it's about FIGHTING DOWNHILL-NESS. It fights against monopolization. Hmm.
//       but the rule precedence in my model allows for that sort of thing to occur. It's just
//       another rule above it in the list that works against the one below it.
//     - save the frames of the discoveries into videos - so you can zip through them fast
//       as so that you have high-quality videos of them all
//     - see if you can compile the images --> video code into webassembly
//     - more data sifting to learn about possibilities and refine "reward function"
//     - node.js script to deploy on many cloud servers and send back seeds?
//     - think about GPU implementation
//     - later: genetic algorithm applied to rules
//     - create levels of abstraction by layering the grids over one another, or something? gravity waves, combined with fuzz, combined with etc. to
//       produce an absurdly complex upper layer (but all the layers interact so you could really view any layer? or do interactions only flow upwards?)

    
    let mut last_seed = 0;
    let mut exploration_count = 0;

    // symbol_count=13, avg_symbols_per_rule=6
    let predefined_seeds_list: Vec::<u64> = vec![];
    let mut predefined_seeds_list_index = 0;
    // ``.trim().split("\n").map(l => l.trim().split(" ").pop()).join(`, `);
    
    let command_line_args: Vec<String> = std::env::args().collect();

    loop {

        let window_size: usize;
        let mut frame_buffer: Vec<u32>;
        let mut window: Window;
        #[cfg(feature="interactive")] {
            window_size = 2usize.pow(10);
            frame_buffer = vec![0; window_size.pow(2)];
            window = Window::new("Emergence", window_size, window_size, WindowOptions::default()).unwrap();
        }

        // symbol_count=13, avg_symbols_per_rule=6
        // water flood: 7467657296677107546
        // large oscillating squares: 14122432294623543467
        // long pink: 6013912011200354030
        // very long icy green: 18032160929456902390
        // long pink teal: 10594241833213785397
        // long faded purple: 2499817319175911526
        // never-ending green-teal waves: 406132538548411765
        // false positive, tiny osscilations forever: 12238960829579750451
        // very cool snaking purple: 16208732140556965857
        // long-lasting wild-fires: 5057473989801057208 , 12872866843233990273
        // repetitive, but there are little circuit things: 14281362661185447652, 5675335502382302640
        // glowing purple lakes: 6660629743018448922
        // cool little "edge circuits": 1863294108166551090
        // very cool: 8402962241731557075
        // very cool: 17744932963446485607
        // cool: 10515637475715549861
        // slowly eating: 18326255651324612896
        // hectic blocky: 765306081269241258  ,  14711061357849914597  ,  13138047963491965255  ,  9687621155272705591
        // interesting/unusual: 17759317169982052868  ,  4374699281681775837  ,  9687621155272705591
        // long-lasting: 15315435961670868935
        // long-lasting blood: 6729337170740254327
        // cells with nucleus: 18097349703590238120
        // FLYING OBJECTS: 906339142304154875
        // very cool & slow & glowing wild-fire: 14742368439561416451 
        // blue crawls, then orange circuits over it: 11934985294093483428
        // QUITE interesting (ever-lasting, but not like the "hectic" ones): 13142931970492573542  ,  7556063397061938808
        // water: 15987640395273323320
        // two stages: 10872786471812694150
        // wormy edges sorta thing: 5978120414185492214
        // COOL MONO-WAVE: 2941821280294775784

        // symbol_count=5, avg_symbols_per_rule=2
        // flying: 15882147986537655481
        // hectic: 2589448072584809481, 2103093424050046521
        // VERY COOL (green line drawers): 10275097085410748598
        // MONO WAVE: 5009945354920515720
        // VERY COOL: 535477901851029657
        // very cool travelling "Front", dying tail: 18131511737090215621
        // oscillations giving rise to larger objects: 8218251321124177500
        // very cool "factory" thing: 16807459843228653455
        // very cool "growing whirlpools" thing: 8269510888484532536

        let seed = if last_seed == 0 { rand::random::<u64>() } else { last_seed };
        //let seed = 5009945354920515720;
        //let seed = if last_seed == 0 { predefined_seeds_list[predefined_seeds_list_index] } else { last_seed }; predefined_seeds_list_index += 1; 

        let size = 2u32.pow(9);
        let symbol_count = 5; // normal=13
        let avg_symbols_per_rule = 4; // remember that there are 9 spaces to match against
        let mut world = World::new(size, symbol_count, avg_symbols_per_rule, seed);
        world.randomize();
        let now = Instant::now();
        let mut count = 0;

        let mut unique_frame_hashes = std::collections::HashSet::new();

        // we compare these at the end and if they're the same then that likely means
        // that the changes are just lots of little repeating oscillations that,
        // when "muliplied" together cause the frames to be unique (thus giving us a false positive).
        let mut last_frames_cell_changes_anded_1 = vec![false; size.pow(2) as usize]; // second last batch of 5 frames, ANDed together
        let mut last_frames_cell_changes_anded_2 = vec![false; size.pow(2) as usize]; // last batch of 5 frames, ANDed together

        let mut already_printed_details = false;
        let print_details = |unique_frame_hashes_len, seed, cell_change_diff_count| println!("unique: {}  cell_change_diff_count: {}  seed: {}", unique_frame_hashes_len, cell_change_diff_count, seed);

        let sample_frame_count = 400;
        let min_end_cell_diff = 25;

        let mut frames = Vec::<Vec<u32>>::new();

        loop {

            world.step();

            // unsafe { println!("scratch_counter_1: {}", scratch_counter_1); }
            // unsafe { println!("scratch_counter_2: {}", scratch_counter_2); }
            // unsafe { println!("scratch_counter_3: {}", scratch_counter_3); }
            unsafe { scratch_counter_1 = 0; }
            unsafe { scratch_counter_2 = 0; }
            unsafe { scratch_counter_3 = 0; }
            //println!("changes: {}", world.data.iter().zip(world.prev_data.iter()).filter(|(a,b)| *a != *b).count());
            
            //world._draw_to_console();
            //std::thread::sleep(Duration::from_millis(1000));

            #[cfg(feature="interactive")] {
                world.draw_to_buffer(&mut frame_buffer);
                window.update_with_buffer(&frame_buffer).unwrap();
                if frames.len() < 1000 { frames.push(world.data.clone()); }
            }

            if count <= sample_frame_count {
                unique_frame_hashes.insert( calculate_vec_hash(&world.data) );
            }
            if count > sample_frame_count-10 && count <= sample_frame_count-5 {
                world.cell_changed_flags.iter().enumerate().for_each(|(i, v)| {
                    if *v {
                        last_frames_cell_changes_anded_1[i] = true;
                    }
                });
            }
            if count > sample_frame_count-5 && count <= sample_frame_count {
                world.cell_changed_flags.iter().enumerate().for_each(|(i, v)| {
                    if *v {
                        last_frames_cell_changes_anded_2[i] = true;
                    }
                });
            }

            #[cfg(feature="interactive")] {
                if window.is_key_down(minifb::Key::Escape) {
                    std::thread::sleep(Duration::from_millis(500));
                    last_seed = 0;
                    break;
                }
                if window.is_key_down(minifb::Key::Enter) {
                    std::thread::sleep(Duration::from_millis(500));
                    last_seed = seed;
                    break;
                }
                if window.is_key_down(minifb::Key::P) {
                    //println!("p key down: {}", window.is_key_down(minifb::Key::P));
                    std::thread::sleep(Duration::from_millis(1000));
                }
                if window.is_key_down(minifb::Key::S) {
                    println!("SAVING GIF");
                    let filename = format!("symbols_{}--seed_{}", symbol_count, seed.to_string());
                    make_gif_from_frames(&frames, &world.symbol_to_color, &filename);
                }

                window.set_title(&count.to_string());
            }
            count += 1;

            let mut there_were_changes = false;
            for changed in world.cell_changed_flags.iter() {
                if *changed {
                    there_were_changes = true;
                    break;
                }
            }

            //if count == 100 { println!("{}", now.elapsed().as_millis()); }
            if count == sample_frame_count || !there_were_changes {
                #[cfg(not(feature="interactive"))] {
                    let cell_change_diff_count = bool_vec_diff_count(&last_frames_cell_changes_anded_1, &last_frames_cell_changes_anded_2);
                    if unique_frame_hashes.len() == sample_frame_count && cell_change_diff_count > min_end_cell_diff {
                        print_details(unique_frame_hashes.len(), seed, cell_change_diff_count);
                    }
                    already_printed_details = true;
                    break;
                }
                #[cfg(feature="interactive")] {
                    if !already_printed_details {
                        let cell_change_diff_count = bool_vec_diff_count(&last_frames_cell_changes_anded_1, &last_frames_cell_changes_anded_2);
                        print_details(unique_frame_hashes.len(), seed, cell_change_diff_count);
                    }
                    already_printed_details = true;
                }
            }
        }

        exploration_count += 1;

        if !already_printed_details {
            let cell_change_diff_count = bool_vec_diff_count(&last_frames_cell_changes_anded_1, &last_frames_cell_changes_anded_2);
            print_details(unique_frame_hashes.len(), seed, cell_change_diff_count);
        }

        if command_line_args.contains(&"benchmark".to_string()) {
            println!("{}", exploration_count);
        }

    }
}

fn bool_vec_diff_count(vec1:&[bool], vec2:&[bool]) -> u32 {
    let mut diff_count = 0;
    for (i, v) in vec1.iter().enumerate() {
        if vec2[i] != *v { diff_count += 1; }
    }
    diff_count
}

fn calculate_vec_hash(vec: &[u32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut s = DefaultHasher::new();
    vec.hash(&mut s);
    s.finish()
}

fn make_gif_from_frames(frames: &[Vec<u32>], colors: &[(u8,u8,u8)], filename:&str) {
    use gif::{Frame, Encoder, Repeat, SetParameter};
    use std::fs::File;
    use std::borrow::Cow;

    let mut flat_colors: Vec<u8> = Vec::with_capacity(colors.len()*3);
    for (r, g, b) in colors.iter() {
        flat_colors.push(*r);
        flat_colors.push(*g);
        flat_colors.push(*b);
    }
    let width = (frames[0].len() as f32).sqrt() as u16;
    let height = width;

    let mut image = File::create(format!("./gifs/{}.gif", filename)).unwrap();
    let mut encoder = Encoder::new(&mut image, width, height, &flat_colors[..]).unwrap();
    encoder.set(Repeat::Infinite).unwrap();
    for frame_data in frames {
        let u8_frame_data: Vec<u8> = frame_data.iter().map(|x| *x as u8).collect();
        let mut frame = Frame::default();
        frame.width = width;
        frame.height = height;
        frame.buffer = Cow::Borrowed(&u8_frame_data[..]);
        encoder.write_frame(&frame).unwrap();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world() {
        //let mut w = World::new(4);
    }
}

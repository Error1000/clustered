// Small tile is 4x4

struct InData {
    inner: u32,
    nrows_left: u32, 
    ncols_right: u32,
    output_matrix_order: u32, // 1 = column major, 2 = row major
    matrix_data: array<mat4x4f>,
}

struct RowMajorMatrix {
    nrows: u32,
    ncols: u32,
    offset: u32
}

struct ColMajorMatrix {
    nrows: u32,
    ncols: u32,
    offset: u32
}


fn get_row_major_offset(i: u32, j: u32, ncols: u32) -> u32 {
    // ncols == number of elements in a row
    return (i*ncols + j);
}

fn get_col_major_offset(i: u32, j: u32, nrows: u32) -> u32 {
    // nrows = number of elements in a column
    return (i + j*nrows);
}

@group(0)
@binding(0)
var<storage, read> in_data: InData;

@group(0)
@binding(1)
var<storage, read_write> out_data: array<mat4x4f>;

@group(0)
@binding(2)
var<uniform> goff: u32;

const BIG_TILE_NROW_LEFT: u32 = u32(1);
const BIG_TILE_INNER_SIZE: u32 = u32(1);
const BIG_TILE_NCOL_RIGHT: u32 = u32(1);
const LEFT_BIG_TILE_NELEM: u32 = BIG_TILE_NROW_LEFT*BIG_TILE_INNER_SIZE;
const RIGHT_BIG_TILE_NELEM: u32 = BIG_TILE_INNER_SIZE*BIG_TILE_NCOL_RIGHT;
const OUTPUT_BIG_TILE_NELEM: u32 = BIG_TILE_NROW_LEFT*BIG_TILE_NCOL_RIGHT;

var<workgroup> left_big_tile: array<mat4x4f, LEFT_BIG_TILE_NELEM>;
var<workgroup> right_big_tile: array<mat4x4f, RIGHT_BIG_TILE_NELEM>;

// Each workgroup computes one big tile
// => the workgroup must do multiple big tile multiplications
// It will load in turn all the left and right big tiles
// then for each one each compute unit calculates in parallel one output small tile
// => it also does multiple small tile multiplications
// Also loading each left and right small tiles in turn and doing element multiplications
// and adding the results.

// What is gained by this is that by loading each left and right tile completly in turn before doing the multiplications of one output tile
// you can avoid having to fetch the elements of the left and right tiles multiple times for each output element.
// Because output element 1 will need the entire first row, but so will output element 2, so
// you can avoid having to load the entire first row twice by loading it all first before
// running the calculations, and because there are multiple levels of cache
// we have multiple levels of tiles,
// the small tiles are for registers, the idea is that you load all of the elements of the small left and right tiles
// into registers then when you compute all the elements of the output tile you avoid several fetches from memory.
// And a similar effect takes place a the workgroup level with workgroup shared memory (i.e core caches)
@compute
@workgroup_size(OUTPUT_BIG_TILE_NELEM)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let actual_id: u32 = gid.x + goff;

    // Deserialise in_data into the 2 big matricies to multiply
    let in1 = RowMajorMatrix(in_data.nrows_left, in_data.inner, 0);

    // This index is expressed in small tiles
    let last_element_of_matrix1_index = get_row_major_offset(in1.nrows-1, in1.ncols-1, in1.ncols)*LEFT_BIG_TILE_NELEM;

    // in1.ncols == in2.nrows, is an implicit assumption of matrix multiplication
    let in2 = RowMajorMatrix(in_data.inner, in_data.ncols_right, last_element_of_matrix1_index+LEFT_BIG_TILE_NELEM);

    let output_nrows: u32 = in1.nrows;
    let output_ncols: u32 = in2.ncols;

    // Each shader invocation calculates one element of the *output* big tile (i.e. one *output* small tile)
    // Each workgroup calculates one element of the *output* matrix (i.e. one *output* big tile)
    // There are a total of BIG_TILE_NROW_LEFT*BIG_TILE_NCOL_RIGHT*output_ncols*output_nrows invocations of the shader
    let wid_i = (actual_id/OUTPUT_BIG_TILE_NELEM)/output_ncols; // row
    let wid_j = (actual_id/OUTPUT_BIG_TILE_NELEM)%output_ncols; // column

    // This specifies which *output* small tile inside the big tile this instance calculates
    let subid_i = (actual_id%OUTPUT_BIG_TILE_NELEM)/BIG_TILE_NCOL_RIGHT;
    let subid_j = (actual_id%OUTPUT_BIG_TILE_NELEM)%BIG_TILE_NCOL_RIGHT;
    let raw_subid = actual_id%OUTPUT_BIG_TILE_NELEM;

    var res = mat4x4f(vec4f(0), vec4f(0), vec4f(0), vec4f(0));
    for(var big_k = u32(0); big_k < in_data.inner; big_k++) {
        // 1. Fetch the two big tiles into workgroup memory
        // The big left tile is at wid_i, big_k
        let big_left_tile_offset = in1.offset + get_row_major_offset(wid_i, big_k, in1.ncols)*LEFT_BIG_TILE_NELEM;

        // The big right tile is at big_k, wid_j
        let big_right_tile_offset = in2.offset + get_row_major_offset(big_k, wid_j, in2.ncols)*RIGHT_BIG_TILE_NELEM;

        if(raw_subid == 0){
            // TODO: Paralellise loading the tiles into workgroup memory
            for(var left_small_tile_i = u32(0); left_small_tile_i < LEFT_BIG_TILE_NELEM; left_small_tile_i++){
                left_big_tile[left_small_tile_i] = in_data.matrix_data[big_left_tile_offset+left_small_tile_i];
            }
            for(var right_small_tile_i = u32(0); right_small_tile_i < RIGHT_BIG_TILE_NELEM; right_small_tile_i++){
                right_big_tile[right_small_tile_i] = in_data.matrix_data[big_right_tile_offset+right_small_tile_i];
            }
        }
                
        // Don't begin computing if some invocations haven't loaded their small tiles yet as the computations for us will rely on the small tiles "of somebody else" 
        workgroupBarrier();

        // 2. Compute result for our element of the big tile ( this uses small_k )
        for (var small_k = u32(0); small_k < BIG_TILE_INNER_SIZE; small_k++){
            let elem_left_offset = get_row_major_offset(subid_i, small_k, BIG_TILE_INNER_SIZE);
            let elem_right_offset = get_row_major_offset(small_k, subid_j, BIG_TILE_NCOL_RIGHT);
            let elem_left = left_big_tile[elem_left_offset]; // In the left matrix
            let elem_right = right_big_tile[elem_right_offset]; // In the right matrix
            res += elem_left*elem_right;
        }


        // Don't start overwriting "our" small tiles if some invocations haven't finished computing yet as the computations of other invocations rely on "our" small tiles
        workgroupBarrier();
    }
    // For debugging:
    // res = mat4x4f(vec4f(f32(subid_i), 0, 0, 0), vec4f(f32(subid_j), 0, 0, 0), vec4f(f32(wid_i), 0, 0, 0), vec4f(f32(wid_j), 0, 0, 0));
    // Add our small tile result to the big *output* tile
    if(in_data.output_matrix_order == 1) {
        out_data[get_row_major_offset(subid_i, subid_j, BIG_TILE_NCOL_RIGHT)+get_col_major_offset(wid_i, wid_j, output_nrows)*OUTPUT_BIG_TILE_NELEM] = res;
    }else if(in_data.output_matrix_order == 2) {
        out_data[get_row_major_offset(subid_i, subid_j, BIG_TILE_NCOL_RIGHT)+get_row_major_offset(wid_i, wid_j, output_ncols)*OUTPUT_BIG_TILE_NELEM] = res;
    }else{
        /*Note: 0xBAD = 2989*/
        out_data[0] = mat4x4f(vec4f(f32(0xBAD), f32(0), f32(0), f32(0)), vec4f(0), vec4f(0), vec4f(0));
    }
}
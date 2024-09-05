struct InData {
    matrix1_ncols: u32,
    matrix1_nrows: u32,
    matrix2_ncols: u32,
    // matrix2_nrows == matrix1_ncols
    output_matrix_order: u32, // 1 = column major, 2 = row major
    matrix_data: array<f32>,
}

struct RowMajorMatrix {
    ncols: u32,
    nrows: u32,
    offset: u32
}

struct ColMajorMatrix {
    ncols: u32,
    nrows: u32,
    offset: u32
}


fn get_row_major_offset(i: u32, j: u32, ncols: u32) -> u32 {
    // ncols == number of elements in a row
    return i*ncols + j;
}

fn get_col_major_offset(i: u32, j: u32, nrows: u32) -> u32 {
    // nrows = number of elements in a column
    return i + j*nrows;
}

@group(0)
@binding(0)
var<storage, read> in_data: InData;

@group(0)
@binding(1)
var<storage, read_write> out_data: array<f32>;

@group(0)
@binding(2)
var<uniform> goff: u32;

const NCHUNKS_PER_ELEM: u32 = 32;
// 32/NCHUNKS_PER_ELEM(chunks) = 1
// Paraphrasing the spec, the values of these atomics are 0:
// WGSL Spec (7.3):
// When a variable in the private, function, or workgroup address spaces is created, it *will* have an initial value. If no initializer is specified the initial value is the default initial value. The initial values are computed as follows:
// If the store type is an atomic type, the zero value is that of the underlying type (concrete integer scalar).
// WGSL Spec (16.1.1):
// The zero values are as follows:
//    u32() is 0u
var<workgroup> elem_locks: array<atomic<u32>, 1>;


@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // goff exists because you can only dispath up to 65536 workgroups in one invocation, and global_invocation_id is only global within one dispatch,
    // and sometimes you may wish to run more instances of the shader than that so the library automatically
    // makes multiple dispatch calls, as many as necessary of the maximum 65536 workgroups and one final one for the remaining number of workgroups
    // to get the amount of workgroups desired by the caller

    // goff = workgroup_size*65536*how_many_dispatched_up_to_this_one, thus goff is a multiple of workgroup_size and gid.x =  workgroup_id * workgroup_size + local_invocation_id. 
    // Then global_instance_id which is = goff + gid.x = workgroup_size*65536*how_many_dispatched_up_to_this_one + workgroup_id * workgroup_size + local_invocation_id
    // Since workgroup_size%NCHUNKS_PER_ELEM = 0
    // Then global_instance_id%NCHUNKS_PER_ELEM = local_invocation_id%NCHUNKS_PER_ELEM
    // And again since workgroup_size%NCHUNKS_PER_ELEM = 0, that means that each workgroup deals with a whole number of elements

    let global_instance_id: u32 = gid.x + goff;

    let elem_to_process_id: u32 = global_instance_id/NCHUNKS_PER_ELEM;
    let lock_id = elem_to_process_id%(32/NCHUNKS_PER_ELEM);
    let chunk_id = global_instance_id%NCHUNKS_PER_ELEM;

    if(elem_to_process_id >= arrayLength(&out_data)) { return; }

    // Deserialise in_data into 2 matricies
    let in1 = RowMajorMatrix(in_data.matrix1_ncols, in_data.matrix1_nrows, 0);
    let last_element_of_matrix1_index = in1.ncols*in1.nrows-1;
    let in2 = ColMajorMatrix(in_data.matrix2_ncols, in_data.matrix1_ncols, last_element_of_matrix1_index+1);
   
    // in1.ncols == in2.nrows
    let output_ncols: u32 = in2.ncols;
    let output_nrows: u32 = in1.nrows;
    

    // Each shader invocation is targeted at one element of the output
    // There are output_ncols elements in a row, i.e. the number of elemens in a row = the number of columns of the matrix
    let id_i = elem_to_process_id/output_ncols; // row
    let id_j = elem_to_process_id%output_ncols; // column


    let k_range_start = in1.ncols/NCHUNKS_PER_ELEM*chunk_id;
    var k_range_end: u32;
    if(chunk_id == NCHUNKS_PER_ELEM-1){
        k_range_end = in1.ncols;
    }else{
        k_range_end = in1.ncols/NCHUNKS_PER_ELEM*(chunk_id+1);
    }

    var chunk_res = 0.0;
    for(var k = k_range_start; k < k_range_end; k++) {
        let elem1_offset = in1.offset + get_row_major_offset(id_i, k, in1.ncols);
        let elem2_offset = in2.offset + get_col_major_offset(k, id_j, in2.nrows);
        let elem1 = in_data.matrix_data[elem1_start_offset]; // In the left matrix
        let elem2 = in_data.matrix_data[elem2_start_offset]; // In the right matrix
        chunk_res += elem1*elem2;
    }
    
    var is_locked = false;
    while(!is_locked){
        let atomic_add_res = atomicAdd(&elem_locks[lock_id], u32(1));
        if atomic_add_res == 0{
            is_locked = true;
            if(in_data.output_matrix_order == 1) {
                out_data[get_col_major_offset(id_i, id_j, output_nrows)] += chunk_res;
            }else if(in_data.output_matrix_order == 2) {
                out_data[get_row_major_offset(id_i, id_j, output_ncols)] += chunk_res;
            }else{
                out_data[0] = f32(0xBAD /*0xBAD = 2989*/);
            }
            atomicStore(&elem_locks[lock_id], u32(0));
        }
    }

}
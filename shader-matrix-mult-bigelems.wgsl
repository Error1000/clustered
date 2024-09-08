struct InData {
    matrix1_ncols: u32,
    matrix1_nrows: u32,
    matrix2_ncols: u32,
    // matrix2_nrows == matrix1_ncols
    output_matrix_order: u32, // 1 = column major, 2 = row major
    matrix_data: array<mat4x4f>,
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


@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let actual_id: u32 = gid.x + goff;
    if(actual_id >= arrayLength(&out_data)) { return; }

    // Deserialise in_data into 2 matricies
    let in1 = ColMajorMatrix(in_data.matrix1_ncols, in_data.matrix1_nrows, 0);
    let last_element_of_matrix1_index = in1.ncols*in1.nrows-1;
    // in1.ncols == in2.nrows, is an implicit assumption of matrix multiplication
    let in2 = RowMajorMatrix(in_data.matrix2_ncols, in1.ncols /* in1.ncols == in2.nrows */, last_element_of_matrix1_index+1);
    let output_ncols: u32 = in2.ncols;
    let output_nrows: u32 = in1.nrows;

    // Each shader invocation calculates one element of the output
    // There are output_ncols elements in a row, i.e. the number of elemens in a row = the number of columns of the matrix
    let id_i = actual_id/output_ncols; // row
    let id_j = actual_id%output_ncols; // column

    let zero_column = vec4f(0, 0, 0, 0);
    var res = mat4x4f(zero_column, zero_column, zero_column, zero_column);
    for(var k = u32(0); k < in1.ncols; k++) {
        let elem1_offset = in1.offset + get_col_major_offset(id_i, k, in1.ncols);
        let elem2_offset = in2.offset + get_row_major_offset(k, id_j, in2.nrows);
        let elem1 = in_data.matrix_data[elem1_offset]; // In the left matrix
        let elem2 = in_data.matrix_data[elem2_offset]; // In the right matrix
        res += elem1*elem2;
    }
    
    if(in_data.output_matrix_order == 1) {
        out_data[get_col_major_offset(id_i, id_j, output_nrows)] = res;
    }else if(in_data.output_matrix_order == 2) {
        out_data[get_row_major_offset(id_i, id_j, output_ncols)] = res;
    }else{
        /*Note: 0xBAD = 2989*/
        out_data[0] = mat4x4f(vec4f(f32(0xBAD), f32(0xBAD), f32(0xBAD), f32(0xBAD)), zero_column, zero_column, zero_column);
    }
}
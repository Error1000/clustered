struct Info {
    input_a_size: u32,
    input_b_size: u32,
    data: array<u32>
}

@group(0)
@binding(0)
var<storage, read> in_data: Info;

@group(0)
@binding(1)
var<storage, read_write> out_data: Info;

@group(0)
@binding(2)
var<uniform> goff: u32;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let actual_id = gid.x+goff;

    var a_size = in_data.input_a_size;
    var b_size = in_data.input_b_size;
    var stride = a_size+b_size;

    var start_offset = actual_id*stride;
    if start_offset >= arrayLength(&in_data.data) {
        return;
    }
    
    if actual_id == 0 {
        out_data.input_a_size = stride;
        out_data.input_b_size = stride;
    }

    var a_start_offset = start_offset;
    var a_end_offset = a_start_offset+a_size-1;
    if a_end_offset >= arrayLength(&in_data.data) {
        a_end_offset = arrayLength(&in_data.data)-1;
        a_size = a_end_offset-a_start_offset+1;
    }

    var b_start_offset = a_end_offset+1;
    var b_end_offset = b_start_offset+b_size-1;
    if b_end_offset >= arrayLength(&in_data.data) {
        b_end_offset = arrayLength(&in_data.data)-1;
        b_size = b_end_offset-b_start_offset+1;
    }

    var input_size = a_size+b_size;
    var end_offset = start_offset+input_size-1;
    if end_offset >= arrayLength(&in_data.data) {
        end_offset = arrayLength(&in_data.data)-1;
        input_size = end_offset-start_offset+1;
    }
    if input_size <= 0 {
        return;
    }

    var out_indx = 0u;
    var a_indx = 0u;
    var b_indx = 0u;

    var a_val = in_data.data[a_start_offset+a_indx];
    var b_val = in_data.data[b_start_offset+b_indx];
    loop {
        if a_val < b_val {
            out_data.data[start_offset+out_indx] = a_val;
            out_indx += 1u;
            a_indx += 1u;
            if a_indx >= a_size { break; }
            a_val = in_data.data[a_start_offset+a_indx];
        }else {
            out_data.data[start_offset+out_indx] = b_val;
            out_indx += 1u;
            b_indx += 1u;
            if b_indx >= b_size { break; }
            b_val = in_data.data[b_start_offset+b_indx];
        }
    }

    while(a_indx < a_size) {
        out_data.data[start_offset+out_indx] = a_val;
        out_indx += 1u;
        a_indx += 1u;
        if a_indx >= a_size { break; }
        a_val = in_data.data[a_start_offset+a_indx];
    }

    while(b_indx < b_size) {
        out_data.data[start_offset+out_indx] = b_val;
        out_indx += 1u;
        b_indx += 1u;
        if b_indx >= b_size { break; }
        b_val = in_data.data[b_start_offset+b_indx];
    }
}
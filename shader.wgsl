@group(0)
@binding(0)
var<storage, read> v_in_data: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> v_out_data: array<f32>;

@group(0)
@binding(2)
var<uniform> goff: u32;

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let actual_id = gid.x+goff;
    if (actual_id >= arrayLength(&v_in_data)){ return; }
    if (actual_id >= arrayLength(&v_out_data)){ return; }
    var e = v_in_data[actual_id];
    for(var i = 0; i < 100; i++){
        e = sqrt(e*e);
    }
    v_out_data[actual_id] = e;
}
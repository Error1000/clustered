@group(0) @binding(0)
var v_in_data: texture_2d<f32>;

@group(0) @binding(1)
var v_out_data: texture_storage_2d<rgba8unorm, write>;

const wsize = vec3<u32>(1, 1, 1);


// Gaussian blur
@compute
@workgroup_size(wsize.x, wsize.y, wsize.z)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var conv_kern: array<array<f32, 5>, 5> = array<array<f32, 5>, 5>(
    array<f32, 5>(1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0),

    array<f32, 5>(4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0),

    array<f32, 5>(7.0/273.0, 26.0/273.0, 41.0/273.0, 26.0/273.0, 7.0/273.0),

    array<f32, 5>(4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0),

    array<f32, 5>(1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0),


    );
    let size = textureDimensions(v_in_data);
    let width = size.x;
    let height = size.y;
    var sum = vec4<f32>(0, 0, 0, 0);
    for(var dx: i32 = -2; dx <= 2; dx++){
        for(var dy: i32 = -2; dy <= 2; dy++){
            let pos_kern = vec2<i32>(2+dx, 2+dy);
            let pos_img = vec2<i32>(dx+i32(gid.x), dy+i32(gid.y));
            var img_val: vec4<f32> = vec4<f32>(0, 0, 0, 0);
            if !(gid.x < 2 || gid.y < 2 || gid.x > width-2 || gid.y > height-2) { 
                img_val = textureLoad(v_in_data, vec2(pos_img.x, pos_img.y), 0);
            }
            if(img_val.w == 0.0){
                img_val = vec4<f32>(0, 0, 0, 0);
            }
            let kern_val: f32 = conv_kern[pos_kern.x][pos_kern.y];
            sum += img_val*kern_val;
        }
    }
    textureStore(v_out_data, vec2(gid.x, gid.y), sum);
}
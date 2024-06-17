use std::borrow::Cow;

use flume::Iter;

pub trait ShaderBytesInfo {
    // NOTE: By *not* taking a self we explicitly disallow dynamically sized types and unsized types
    // Because working with consistently sized types is overall better (opinion)
    // even if those types then need to contain pointers to dynamically sized data
    fn shader_bytes_size() -> usize;
    fn shader_bytes_align() -> usize;
}

/// # Safety
/// Implementor must guarantee that the result is correct
/// for the shader memory layout (See std140 and std430 for details)
pub unsafe trait IntoShaderBytes: ShaderBytesInfo {
    fn to_shader_bytes(&self, res: &mut [u8]);
}

/// # Safety
/// Implementor must guarantee that the result is correctly read
/// from the shader result
pub unsafe trait FromShaderBytes: ShaderBytesInfo {
    fn from_shader_bytes(buf: &[u8]) -> Self;
}

impl ShaderBytesInfo for u32 {
    fn shader_bytes_size() -> usize {
        core::mem::size_of::<Self>()
    }
    fn shader_bytes_align() -> usize {
        core::mem::size_of::<Self>()
    }
}

// Source for alignment, sizes and endianness: https://www.w3.org/TR/WGSL/#memory-layouts
// Note: While this theoretically only applies to wgsl, i'm using it as the defacto standard since it seems like the only one that is well-defined
// TODO: In the future if problems arise it may be nice to dynamically test for differences like endianness or god forbid two's complement vs other formats like one's complement
//       and adjust these functions dynamically to do the correct thing.
//       Although i'm not sure how you would test for endianness at the spir-v level since it seems to explicitly disallow any operations which could depends on the endianness.
// TODO: Do more research in the question of endianness above.

unsafe impl IntoShaderBytes for u32 {
    fn to_shader_bytes(&self, res: &mut [u8]) {
        for (i, e) in self.to_le_bytes().iter().enumerate() {
            res[i] = *e;
        }
    }
}

unsafe impl FromShaderBytes for u32 {
    fn from_shader_bytes(buf: &[u8]) -> Self {
        Self::from_le_bytes(buf.try_into().unwrap())
    }
}

impl ShaderBytesInfo for i32 {
    fn shader_bytes_size() -> usize {
        core::mem::size_of::<Self>()
    }
    fn shader_bytes_align() -> usize {
        core::mem::size_of::<Self>()
    }
}

unsafe impl IntoShaderBytes for i32 {
    fn to_shader_bytes(&self, res: &mut [u8]) {
        for (i, e) in self.to_le_bytes().iter().enumerate() {
            res[i] = *e;
        }
    }
}

unsafe impl FromShaderBytes for i32 {
    fn from_shader_bytes(buf: &[u8]) -> Self {
        Self::from_le_bytes(buf.try_into().unwrap())
    }
}

impl ShaderBytesInfo for f32 {
    fn shader_bytes_size() -> usize {
        core::mem::size_of::<Self>()
    }
    fn shader_bytes_align() -> usize {
        core::mem::size_of::<Self>()
    }
}

unsafe impl IntoShaderBytes for f32 {
    fn to_shader_bytes(&self, res: &mut [u8]) {
        for (i, e) in self.to_le_bytes().iter().enumerate() {
            res[i] = *e;
        }
    }
}

unsafe impl FromShaderBytes for f32 {
    fn from_shader_bytes(buf: &[u8]) -> Self {
        Self::from_le_bytes(buf.try_into().unwrap())
    }
}

pub struct ShaderBytes<'a> {
    inner: Cow<'a, [u8]>,
}

impl<'a> ShaderBytes<'a> {
    pub fn get_data(&self) -> &[u8] {
        &self.inner
    }

    pub fn into_data(self) -> Cow<'a, [u8]> {
        self.inner
    }

    /// # Safety
    /// Called must guarantee that the data is in the right format for the shader
    /// That is, memory layout must be correct (run_shader uses storage buffers so expects std430)
    pub unsafe fn from_raw(data: &[u8]) -> ShaderBytes {
        ShaderBytes {
            inner: Cow::from(data),
        }
    }

    pub fn serialise_from_slice<T>(data: &[T]) -> ShaderBytes
    where
        T: IntoShaderBytes,
    {
        let stride: usize =
            usize::next_multiple_of(T::shader_bytes_size(), T::shader_bytes_align());
        let mut serialised = vec![0u8; data.len() * stride];
        for (i, raw_bytes) in serialised.chunks_exact_mut(stride).enumerate() {
            T::to_shader_bytes(&data[i], raw_bytes);
        }

        ShaderBytes {
            inner: Cow::from(serialised),
        }
    }

    pub fn deserialise_to_slice<T>(data: &[u8]) -> impl Iterator<Item = T> + '_
    where
        T: FromShaderBytes,
    {
        let stride: usize =
            usize::next_multiple_of(T::shader_bytes_size(), T::shader_bytes_align());
        data.chunks_exact(stride)
            .map(|raw_bytes| T::from_shader_bytes(raw_bytes))
    }
}

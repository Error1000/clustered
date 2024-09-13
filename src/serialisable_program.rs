use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_with::{base64::Base64, serde_as};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ShaderModuleDescriptor,
};

#[serde_as]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerialisableProgram {
    #[serde_as(as = "Base64")]
    pub in_data: Vec<u8>,
    pub out_data_nbytes: usize,
    pub program: String,
    pub entry_point: String,
    pub n_workgroups: usize,
    pub workgroup_size: usize,
}

impl SerialisableProgram {
    pub async fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Vec<u8>> {
        let cm = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::from(&self.program)),
        });
        let in_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &self.in_data,
            usage: BufferUsages::STORAGE,
        });

        let mut out_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: self.out_data_nbytes.try_into().unwrap(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        crate::run_shader(crate::RunShaderParams {
            device,
            queue,
            in_buf: &in_buf,
            out_buf: &mut out_buf,
            workgroup_len: self.workgroup_size,
            n_workgroups: self.n_workgroups,
            program: &cm,
            entry_point: &self.entry_point,
        })?;

        let transfer_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: out_buf.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&out_buf, 0, &transfer_buf, 0, out_buf.size());
        queue.submit([enc.finish()].into_iter());

        let transfer_view = transfer_buf.slice(..);
        crate::wgpu_map_helper(device, wgpu::MapMode::Read, &transfer_view)
            .await
            .ok()?;
        let res = transfer_view
            .get_mapped_range()
            .iter()
            .copied()
            .collect::<Vec<u8>>();
        Some(res)
    }
}

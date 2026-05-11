#!/usr/bin/env python3
import os

# 定义文件合并顺序
file_order = [
    'vllm_code_structure_analysis.md',
    '01_architecture_overview.md',
    '02_config_system.md',
    '03_engine_core.md',
    '04_scheduler.md',
    '05_pagedattention.md',
    '06_attention_backends.md',
    '07_model_executor.md',
    '08_worker_and_executor.md',
    '09_kv_cache.md',
    '10_sampling.md',
    '11_multimodal.md',
    '12_quantization.md',
    '13_distributed.md',
    '14_lora.md',
    '15_api_layer.md',
    '16_cuda_kernels.md',
    '17_compilation_and_optimization.md',
    '18_structured_output.md',
    '19_speculative_decode.md',
    '20_metrics_and_monitoring.md',
    '21_design_patterns.md'
]

output_file = 'vllm_complete_guide.md'
readcode_dir = '/workspace/ReadCode'

# 合并文件
with open(os.path.join(readcode_dir, output_file), 'w', encoding='utf-8') as outfile:
    for filename in file_order:
        file_path = os.path.join(readcode_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(f'\n<!-- ============================================ -->')
                outfile.write(f'\n<!-- 开始: {filename} -->')
                outfile.write(f'\n<!-- ============================================ -->')
                outfile.write(f'\n\n')
                outfile.write(infile.read())
                outfile.write(f'\n\n')
                outfile.write(f'<!-- ============================================ -->')
                outfile.write(f'\n<!-- 结束: {filename} -->')
                outfile.write(f'\n<!-- ============================================ -->')
                outfile.write(f'\n\n')
            print(f'已合并: {filename}')
        else:
            print(f'警告: 文件不存在 {filename}')

print(f'\n合并完成！输出文件: {output_file}')

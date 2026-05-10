const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8081;
const DOCS_DIR = '/workspace/ReadCode';

const FILES = fs.readdirSync(DOCS_DIR)
  .filter(f => f.endsWith('.md'))
  .sort((a, b) => {
    const order = ['vllm_code_structure_analysis.md',
      '01_architecture_overview.md', '02_config_system.md', '03_engine_core.md',
      '04_scheduler.md', '05_pagedattention.md', '06_attention_backends.md',
      '07_model_executor.md', '08_worker_and_executor.md', '09_kv_cache.md',
      '10_sampling.md', '11_multimodal.md', '12_quantization.md',
      '13_distributed.md', '14_lora.md', '15_api_layer.md', '16_cuda_kernels.md',
      '17_compilation_and_optimization.md', '18_structured_output.md',
      '19_speculative_decode.md', '20_metrics_and_monitoring.md',
      '21_design_patterns.md', 'test_mermaid.md'];
    return order.indexOf(a) - order.indexOf(b);
  });

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function buildIndexPage() {
  const items = FILES.map(f => {
    const name = f.replace('.md', '');
    const title = name === 'vllm_code_structure_analysis.md' ? f : name
      .replace(/^\d+_/, '')
      .replace(/_/g, ' ');
    return `<li><a href="/view?f=${encodeURIComponent(f)}"><strong>${escapeHtml(name)}</strong></a> — ${title}</li>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>vLLM 深度分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif; background:#0f1117; color:#e6edf3; line-height:1.7; }
  h1 { text-align:center; padding:40px 20px 20px; font-size:28px; background:linear-gradient(135deg, #1a1f35, #16213e); color:#58a6ff; border-bottom:1px solid #30363d; }
  h1 span { font-size:14px; color:#8b949e; display:block; margin-top:8px; font-weight:400; }
  .stats { text-align:center; color:#8b949e; padding:10px; font-size:13px; }
  .container { max-width:1100px; margin:0 auto; padding:30px 20px; }
  ul { list-style:none; }
  li { background:#161b22; border:1px solid #30363d; border-radius:8px; margin:10px 0; padding:16px 20px; transition:all .2s; }
  li:hover { border-color:#58a6ff; transform:translateX(4px); }
  li a { color:#e6edf3; text-decoration:none; display:block; }
  li strong { color:#58a6ff; font-size:15px; }
  footer { text-align:center; padding:30px; color:#484f58; font-size:12px; border-top:1px solid #21262d; margin-top:40px; }
</style>
</head>
<body>
<h1>vLLM 深度分析报告 <span>22 个文档 · 约 28,000 行</span></h1>
<div class="stats">共 ${FILES.length} 篇文档 | Mermaid 架构图已内嵌渲染</div>
<div class="container"><ul>${items}</ul></div>
<footer>基于 vLLM main 分支源码生成</footer>
<script>
mermaid.initialize({ 
  startOnLoad: true, 
  theme: 'dark',
  themeVariables: {
    primaryColor: '#58a6ff',
    primaryTextColor: '#e6edf3',
    lineColor: '#30363d',
    secondaryColor: '#161b22',
    tertiaryColor: '#1f2937'
  }
});
</script>
</body></html>`;
}

function buildViewerPage(filename, content) {
  const encodedFile = encodeURIComponent(filename);
  const fileIndex = FILES.indexOf(filename);
  const prevFile = fileIndex > 0 ? FILES[fileIndex - 1] : null;
  const nextFile = fileIndex < FILES.length - 1 ? FILES[fileIndex + 1] : null;

  const navLinks = [];
  if (prevFile) navLinks.push(`<a href="/view?f=${encodeURIComponent(prevFile)}">&larr; ${prevFile}</a>`);
  navLinks.push(`<a href="/">🏠 首页</a>`);
  if (nextFile) navLinks.push(`<a href="/view?f=${encodeURIComponent(nextFile)}">${nextFile} &rarr;</a>`);

  // 简单处理，直接把文档内容放入，用 marked 的风格处理一下
  let html = content
    // 处理 mermaid 先
    .replace(/```mermaid([\s\S]*?)```/g, (match, code) => {
      return `<div class="mermaid">${code}</div>`;
    })
    // 处理其他代码块
    .replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code)}</code></pre>`;
    })
    // 标题
    .replace(/^### (.*)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*)$/gm, '<h2>$1</h2>')
    .replace(/^# (.*)$/gm, '<h1>$1</h1>')
    // 加粗和代码
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code class="inline">$1</code>')
    // 换行和段落
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(filename)} — vLLM 深度分析</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif; background:#0f1117; color:#e6edf3; line-height:1.75; }
  .navbar { position:sticky; top:0; z-index:100; background:#161b22; border-bottom:1px solid #30363d; padding:12px 24px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  .navbar a { color:#58a6ff; text-decoration:none; font-size:14px; padding:6px 12px; border-radius:6px; border:1px solid #30363d; white-space:nowrap; }
  .navbar a:hover { background:#1f2937; border-color:#58a6ff; }
  .file-title { color:#f0883e; font-weight:700; font-size:18px; margin-right:auto; }
  .container { max-width:1200px; margin:0 auto; padding:30px 40px; }
  h1 { color:#f0883e; font-size:26px; margin:30px 0 16px; padding-bottom:10px; border-bottom:2px solid #30363d; }
  h2 { color:#79c0ff; font-size:21px; margin:28px 0 14px; padding-left:12px; border-left:4px solid #58a6ff; }
  h3 { color:#d2a8ff; font-size:17px; margin:22px 0 10px; }
  p { margin:10px 0; color:#c9d1d9; }
  strong { color:#ffa657; }
  code.inline { background:#1f2937; color:#ff7b72; padding:2px 6px; border-radius:4px; font-size:88%; }
  pre { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; overflow-x:auto; margin:14px 0; }
  pre code { color:#c9d1d9; font-size:13px; line-height:1.6; }
  table { width:100%; border-collapse:collapse; margin:16px 0; font-size:14px; }
  th { background:#1f2937; color:#ffa657; padding:10px 14px; text-align:left; border:1px solid #30363d; }
  td { padding:9px 14px; border:1px solid #21262d; }
  tr:nth-child(even) { background:#161b22; }
  ul { padding-left:24px; margin:10px 0; list-style:disc; }
  li { margin:5px 0; color:#c9d1d9; }
  .mermaid { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin:18px 0; display:flex; justify-content:center; overflow-x:auto; }
  .mermaid svg { max-width:100%; height:auto; }
  blockquote { border-left:4px solid #58a6ff; background:#161b22; padding:12px 20px; margin:14px 0; border-radius:0 8px 8px 0; color:#8b949e; }
</style>
</head>
<body>
<div class="navbar">
  <span class="file-title">${escapeHtml(filename)}</span>
  ${navLinks.join('')}
</div>
<div class="container">${html}</div>
<script>
mermaid.initialize({ 
  startOnLoad: true, 
  theme: 'dark',
  securityLevel: 'loose',
  themeVariables: {
    primaryColor: '#58a6ff',
    primaryTextColor: '#e6edf3',
    lineColor: '#30363d',
    secondaryColor: '#161b22',
    tertiaryColor: '#1f2937'
  }
});
</script>
</body></html>`;
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  if (url.pathname === '/' || url.pathname === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(buildIndexPage());
    return;
  }

  if (url.pathname === '/view') {
    const f = url.searchParams.get('f');
    if (!f || !FILES.includes(f)) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('File not found');
      return;
    }
    const content = fs.readFileSync(path.join(DOCS_DIR, f), 'utf-8');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(buildViewerPage(f, content));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`vLLM 文档预览服务已启动!`);
  console.log(`  本地访问: http://localhost:${PORT}`);
  console.log(`  共 ${FILES.length} 个文档可浏览`);
});

import{_ as s,c as a,d as p,o as t}from"./app--uQ_sZr1.js";const e={};function o(c,n){return t(),a("div",null,n[0]||(n[0]=[p(`<h1>conv2d_transpose</h1><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token keyword">fn</span> <span class="token function-definition function">conv2d_transpose</span><span class="token punctuation">(</span></span>
<span class="line">    x<span class="token punctuation">:</span> <span class="token operator">&amp;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token operator">&gt;</span><span class="token punctuation">,</span></span>
<span class="line">    kernels<span class="token punctuation">:</span> <span class="token operator">&amp;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token operator">&gt;</span><span class="token punctuation">,</span></span>
<span class="line">    steps<span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token keyword">i64</span><span class="token punctuation">;</span> <span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    padding<span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token punctuation">(</span><span class="token keyword">i64</span><span class="token punctuation">,</span> <span class="token keyword">i64</span><span class="token punctuation">)</span><span class="token punctuation">;</span> <span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    output_padding<span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token keyword">i64</span><span class="token punctuation">;</span> <span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    dilation<span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token keyword">i64</span><span class="token punctuation">;</span> <span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line"><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token operator">&gt;</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Performs a 2D convolution operation with support for stride, padding, dilation, and activation functions.</p><h2>Parameters:</h2><p><code>x</code>: Input tensor with shape [batch_size, height, width, in_channels]</p><p><code>kernels</code>: Transposed convolution kernels tensor with shape [kernel_height, kernel_width, out_channels, in_channels]</p><p><code>steps</code>: Convolution stride as [step_height, step_width]</p><p><code>padding</code>: Padding size as [(padding_top, padding_bottom), (padding_left, padding_right)]</p><p><code>dilation</code>: Kernel dilation factors as [dilation_height, dilation_width]</p><h2>Returns:</h2><p>Tensor with type <code>T</code></p><h2>Examples:</h2><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token keyword">use</span> <span class="token namespace">hpt<span class="token punctuation">::</span></span><span class="token punctuation">{</span><span class="token class-name">Conv</span><span class="token punctuation">,</span> <span class="token class-name">Random</span><span class="token punctuation">,</span> <span class="token class-name">Tensor</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token punctuation">,</span> <span class="token class-name">TensorInfo</span><span class="token punctuation">}</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">fn</span> <span class="token function-definition function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token comment">// [batch_size, height, width, in_channels]</span></span>
<span class="line">    <span class="token keyword">let</span> input <span class="token operator">=</span> <span class="token class-name">Tensor</span><span class="token punctuation">::</span><span class="token operator">&lt;</span><span class="token keyword">f32</span><span class="token operator">&gt;</span><span class="token punctuation">::</span><span class="token function">randn</span><span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">16</span><span class="token punctuation">,</span> <span class="token number">16</span><span class="token punctuation">,</span> <span class="token number">32</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line">    <span class="token comment">// [kernel_height, kernel_width, out_channels, in_channels]</span></span>
<span class="line">    <span class="token keyword">let</span> kernels <span class="token operator">=</span> <span class="token class-name">Tensor</span><span class="token punctuation">::</span><span class="token operator">&lt;</span><span class="token keyword">f32</span><span class="token operator">&gt;</span><span class="token punctuation">::</span><span class="token function">randn</span><span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">16</span><span class="token punctuation">,</span> <span class="token number">32</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line">    <span class="token comment">// Perform transposed convolution to upsample the feature map</span></span>
<span class="line">    <span class="token keyword">let</span> output <span class="token operator">=</span> input<span class="token punctuation">.</span><span class="token function">conv2d_transpose</span><span class="token punctuation">(</span></span>
<span class="line">        <span class="token operator">&amp;</span>kernels<span class="token punctuation">,</span></span>
<span class="line">        <span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">,</span>           <span class="token comment">// stride</span></span>
<span class="line">        <span class="token punctuation">[</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token comment">// padding</span></span>
<span class="line">        <span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span>           <span class="token comment">// output_padding</span></span>
<span class="line">        <span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span>           <span class="token comment">// dilation</span></span>
<span class="line">    <span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line">    <span class="token macro property">println!</span><span class="token punctuation">(</span><span class="token string">&quot;Output shape: {:?}&quot;</span><span class="token punctuation">,</span> output<span class="token punctuation">.</span><span class="token function">shape</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span> <span class="token comment">// [1, 31, 31, 16]</span></span>
<span class="line">    <span class="token class-name">Ok</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2>Backend Support</h2><table><thead><tr><th>Backend</th><th>Supported</th></tr></thead><tbody><tr><td>CPU</td><td>✅</td></tr><tr><td>Cuda</td><td>❌</td></tr></tbody></table>`,15)]))}const i=s(e,[["render",o],["__file","conv2d_transpose.html.vue"]]),u=JSON.parse('{"path":"/user_guide/conv/conv2d_transpose.html","title":"conv2d_transpose","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"Parameters:","slug":"parameters","link":"#parameters","children":[]},{"level":2,"title":"Returns:","slug":"returns","link":"#returns","children":[]},{"level":2,"title":"Examples:","slug":"examples","link":"#examples","children":[]},{"level":2,"title":"Backend Support","slug":"backend-support","link":"#backend-support","children":[]}],"git":{"updatedTime":1741198455000,"contributors":[{"name":"Jianqoq","username":"Jianqoq","email":"120760306+Jianqoq@users.noreply.github.com","commits":1,"url":"https://github.com/Jianqoq"}]},"filePathRelative":"user_guide/conv/conv2d_transpose.md"}');export{i as comp,u as data};

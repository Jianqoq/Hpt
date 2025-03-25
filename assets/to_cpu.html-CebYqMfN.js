import{_ as n,c as a,d as t,o as e}from"./app-g8MHUUnK.js";const p={};function o(c,s){return e(),a("div",null,s[0]||(s[0]=[t(`<h1>to_cpu</h1><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token keyword">fn</span> <span class="token function-definition function">to_cpu</span><span class="token operator">&lt;</span><span class="token keyword">const</span> <span class="token constant">DEVICE_ID</span><span class="token punctuation">:</span> <span class="token keyword">usize</span><span class="token operator">&gt;</span><span class="token punctuation">(</span>x<span class="token punctuation">:</span> <span class="token operator">&amp;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token punctuation">,</span> <span class="token class-name">Cuda</span><span class="token operator">&gt;</span><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token punctuation">,</span> <span class="token class-name">Cpu</span><span class="token operator">&gt;</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><p>Transfers a tensor from CUDA GPU memory to CPU memory, creating a new tensor in host memory.</p><p>currently only <code>DEVICE_ID</code> = 0 is supported</p><h2>Parameters:</h2><p><code>DEVICE_ID</code>: A compile-time constant specifying the target CPU device ID (default is 0)</p><h2>Returns:</h2><p>A new <code>Tensor&lt;T&gt;</code> located on the specified CPU device, or a TensorError if the transfer fails.</p><h2>Examples:</h2><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token keyword">use</span> <span class="token namespace">hpt<span class="token punctuation">::</span></span><span class="token punctuation">{</span><span class="token namespace">backend<span class="token punctuation">::</span></span><span class="token class-name">Cuda</span><span class="token punctuation">,</span> <span class="token namespace">error<span class="token punctuation">::</span></span><span class="token class-name">TensorError</span><span class="token punctuation">,</span> <span class="token namespace">ops<span class="token punctuation">::</span></span><span class="token class-name">TensorCreator</span><span class="token punctuation">,</span> <span class="token class-name">Tensor</span><span class="token punctuation">}</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">fn</span> <span class="token function-definition function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token keyword">let</span> a <span class="token operator">=</span> <span class="token class-name">Tensor</span><span class="token punctuation">::</span><span class="token operator">&lt;</span><span class="token keyword">f32</span><span class="token punctuation">,</span> <span class="token class-name">Cuda</span><span class="token operator">&gt;</span><span class="token punctuation">::</span><span class="token function">empty</span><span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token macro property">println!</span><span class="token punctuation">(</span><span class="token string">&quot;{}&quot;</span><span class="token punctuation">,</span> a<span class="token punctuation">.</span><span class="token function">to_cpu</span><span class="token punctuation">::</span><span class="token operator">&lt;</span><span class="token number">0</span><span class="token operator">&gt;</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">)</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token class-name">Ok</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,10)]))}const u=n(p,[["render",o],["__file","to_cpu.html.vue"]]),r=JSON.parse('{"path":"/user_guide/associated_methods/cuda/to_cpu.html","title":"to_cpu","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"Parameters:","slug":"parameters","link":"#parameters","children":[]},{"level":2,"title":"Returns:","slug":"returns","link":"#returns","children":[]},{"level":2,"title":"Examples:","slug":"examples","link":"#examples","children":[]}],"git":{"updatedTime":1742936099000,"contributors":[{"name":"Jianqoq","username":"Jianqoq","email":"120760306+Jianqoq@users.noreply.github.com","commits":1,"url":"https://github.com/Jianqoq"}]},"filePathRelative":"user_guide/associated_methods/cuda/to_cpu.md"}');export{u as comp,r as data};

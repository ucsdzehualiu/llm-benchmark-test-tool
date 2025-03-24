import argparse
import asyncio
import random
import statistics
import time
from operator import truediv

from openai import AsyncOpenAI

#vllm serve mistralai/Mistral-Nemo-Instruct-2407 --swap-space 16 --tensor_parallel_size 1 --gpu-memory-utilization 0.98 --disable-log-requests
class AsyncLLMBenchmark:
    def __init__(self, base_url, model, num_requests, concurrency, max_tokens, temperature, min_chars, max_chars):
        self.client = AsyncOpenAI(base_url=base_url, api_key="none")
        self.model = model
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.results = []
        self.total_duration = 0
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _async_stream_request(self, messages):
        """处理流式响应"""

        try:
            stream = await self.client.chat.completions.create(  # 改为chat.completions
                model=self.model,
                messages=messages,  # 使用正确的messages参数
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                stop='<OO>'
            )
            start_time = time.monotonic()
            first_token_time = None
            full_response = []

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content and first_token_time is None:
                        first_token_time =  time.monotonic()
                    if delta_content:
                        full_response.append(delta_content)
                if chunk.choices[0].finish_reason is not None:
                    print(chunk.choices[0].finish_reason)
                    break
                        # output_tokens += 1
            # async for chunk in stream:
            #     current_time = time.monotonic()
            #     if chunk.choices[0].text:
            #         if first_token_time is None:
            #             first_token_time = current_time
            #         last_token_time = current_time
            #         output_tokens += 1
            end_time = time.monotonic()
            output_tokens=len(full_response)
            input_tokens = sum(len(msg["content"]) for msg in messages)
            output_duration = end_time - first_token_time if first_token_time else 0
            print(f"prompt 长度: {input_tokens}，生成的 output 长度 {output_tokens}")

            return {
                'ttft': first_token_time - start_time if first_token_time else None,
                'tpot': output_duration / output_tokens if output_tokens > 0 else None,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'start': start_time,
                'end': end_time,
                'output_tokens_per_sec': output_tokens / output_duration if output_duration > 0 else 0
            }

        except Exception as e:
            print(f"Request error: {str(e)}")
            return None

    def _generate_messages(self):
        """生成强制长回复的提示结构"""
        # 基础字符长度扩展
        char_count = random.randint(self.min_chars, self.max_chars)

        # 结构化模板库
        templates = [
            {
                "type": "学术论文",
                "structure": [
                    "请撰写一篇不少于3000字的学术论文，题目为《{topic}》。要求包含：",
                    "1. 研究背景（至少500字，需引用3篇以上参考文献）",
                    "2. 方法论（详细描述实验设计和实施步骤）",
                    "3. 数据分析（包含图表解读和统计检验）",
                    "4. 讨论部分（比较现有研究成果并分析局限性）",
                    "5. 结论与展望（提出至少三个未来研究方向）",
                    "请确保每个章节完整且达到字数要求，不要提前结束。"
                ],
                "topics": [
                    "深度学习在蛋白质结构预测中的应用",
                    "量子计算对现代密码学的影响",
                    "火星殖民计划的生态闭环系统设计"
                ]
            },
            {
                "type": "技术文档",
                "structure": [
                    "请编写完整的技术文档，主题：{topic}。文档需包含：",
                    "## 1. 核心原理（从数学公式推导开始）",
                    "## 2. 系统架构图及组件说明",
                    "## 3. 部署流程（含Kubernetes集群配置示例）",
                    "## 4. 性能优化（至少提供5种调优策略）",
                    "## 5. 故障排查手册（常见错误代码表）",
                    "文档总长度需超过4000字，每个章节必须完整。"
                ],
                "topics": [
                    "基于Transformer的大规模分布式训练系统",
                    "实时风控系统的流式处理架构",
                    "多模态LLM的服务化部署方案"
                ]
            },
            {
                "type": "分析报告",
                "structure": [
                    "请生成完整的行业分析报告：{topic}。结构要求：",
                    "Ⅰ. 市场现状（数据需包含近5年统计）",
                    "Ⅱ. 技术路线对比（列表比较至少10项参数）",
                    "Ⅲ. 产业链图谱（上游供应商到下游应用场景）",
                    "Ⅳ. 风险评估（SWOT+PESTEL分析）",
                    "Ⅴ. 投资建议（分短期/中期/长期策略）",
                    "每个部分不少于800字，总报告需超过5000字。"
                ],
                "topics": [
                    "2024-2030年人工智能芯片行业发展预测",
                    "可控核聚变商业化路径分析",
                    "脑机接口技术的医疗应用前景"
                ]
            }
        ]

        # 随机选择模板
        template = random.choice(templates)
        topic = random.choice(template["topics"])

        # 构建基础提示
        prompt_lines = [line.format(topic=topic) for line in template["structure"]]
        base_prompt = "\n".join(prompt_lines)

        # 添加防截断指令
        anti_truncation = [
            "\n重要要求：",
            "1. 请务必生成完整内容，不要提前结束",
            "2. 每个章节必须达到指定字数",
            "3. 如果需要更多空间可以继续扩展",
            "4. 避免使用'以下简略说明'等缩短性表述",
            "5. 保持技术细节的完整性"
        ]

        # 组合最终提示
        final_prompt = base_prompt + "\n" + "\n".join(anti_truncation)

        # 长度调整策略
        while len(final_prompt) < char_count:
            expansion_phrases = [
                "\n补充说明：需要特别强调的是......",
                "\n扩展分析：从另一个视角来看......",
                "\n技术细节补充：具体来说......",
                "\n历史背景追溯：早在20世纪......",
                "\n对比研究：与传统方法相比......",
                "\n典型案例：例如在2023年的......"
            ]
            final_prompt += random.choice(expansion_phrases)
            final_prompt += " " * random.randint(50, 100)  # 添加空白延长

        # 确保最终长度
        final_prompt = final_prompt[:char_count].rstrip() + "..."  # 保持完整性

        return [{"role": "user", "content": final_prompt}]
    # def _generate_messages(self):
    #     """生成符合completions接口的prompt"""
    #     char_count = random.randint(self.min_chars, self.max_chars)
    #     # 生成纯文本提示（不带chat格式）
    #     return ''.join(random.choice('模型性能测试验证') for _ in range(char_count))

    async def _worker(self):
        async with self.semaphore:
            messages = self._generate_messages()
            result = await self._async_stream_request(messages)
            if result and result['ttft'] is not None:
                self.results.append(result)



    async def run(self):
        start_time = time.monotonic()
        tasks = [self._worker() for _ in range(self.num_requests)]
        await asyncio.gather(*tasks)
        self.total_duration = time.monotonic() - start_time

    def generate_report(self):
        if not self.results:
            return None
        valid_results = [r for r in self.results if r['ttft'] is not None]
        try:
            ttft = [r['ttft'] for r in valid_results]
            tpot = [r['tpot'] for r in valid_results if r['tpot'] is not None]
            output_speed = [r['output_tokens_per_sec'] for r in valid_results]

            input_tokens = sum(r['input_tokens'] for r in self.results)
            output_tokens = sum(r['output_tokens'] for r in self.results)
            # for r in self.results:
            #     print(r['output_tokens'])

            report = {
                'concurrency': self.concurrency,
                'success': len(self.results),
                'total_requests': self.num_requests,
                'total_duration': self.total_duration,
                'ttft_mean': statistics.mean(ttft) if ttft else 0,
                'ttft_stdev': statistics.stdev(ttft) if len(ttft) > 1 else 0,
                'ttft_p95': statistics.quantiles(ttft, n=100)[94] if len(ttft) > 1 else 0,
                'tpot_mean': statistics.mean(tpot) if tpot else 0,
                'tpot_stdev': statistics.stdev(tpot) if len(tpot) > 1 else 0,
                'tpot_p95': statistics.quantiles(tpot, n=100)[94] if len(tpot) > 1 else 0,
                'input_tokens_per_sec': input_tokens / self.total_duration if self.total_duration > 0 else 0,
                'output_tokens_per_sec_total': output_tokens / self.total_duration if self.total_duration > 0 else 0,
                'rpm': len(self.results) / self.total_duration * 60 if self.total_duration > 0 else 0,
                'output_tokens_per_sec_avg': statistics.mean(output_speed) if output_speed else 0,

            }
            return report
        except statistics.StatisticsError as e:
            print(f"统计错误: {str(e)}")
            return None



async def main(args):
    benchmark = AsyncLLMBenchmark(
        base_url=args.base_url,
        model=args.model,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        min_chars=args.min_chars,
        max_chars=args.max_chars
    )
    await benchmark.run()
    report = benchmark.generate_report()
    if report:
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Async LLM Benchmark')
    parser.add_argument('--base-url', type=str, required=True, help='API 的 Base URL')
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--num-requests', type=int, required=True, help='请求总数')
    parser.add_argument('--concurrency', type=int, required=True, help='并发数')
    parser.add_argument('--max_tokens', type=int, default=512, help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--min_chars', type=int, default=1200, help='生成消息的最小字符数')
    parser.add_argument('--max_chars', type=int, default=1300, help='生成消息的最大字符数')
    args = parser.parse_args()

    asyncio.run(main(args))

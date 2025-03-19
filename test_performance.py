import asyncio
import json
import time
import random
import statistics
from openai import AsyncOpenAI


class AsyncLLMBenchmark:
    def __init__(self, base_url, model, num_requests, concurrency):
        self.client = AsyncOpenAI(base_url=base_url, api_key="none")
        self.model = model
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.results = []
        self.total_duration = 0
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _async_stream_request(self, messages):
        """处理流式响应"""
        start_time = time.monotonic()
        first_token_time = None
        last_token_time = start_time
        full_response = []

        try:
            # 使用chat.completions接口
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                stream=True,
                stop=['<|im_end|>']
            )

            async for chunk in stream:
                current_time = time.monotonic()
                delta_content = chunk.choices[0].delta.content or ""

                # 记录第一个token时间
                if delta_content and first_token_time is None:
                    first_token_time = current_time

                # 更新最后一个token时间
                if delta_content:
                    last_token_time = current_time
                    full_response.append(delta_content)

            end_time = time.monotonic()

            # 计算token数量（更精确的实现可能需要分词器）
            input_tokens = sum(len(msg["content"]) for msg in messages)
            output_tokens = len("".join(full_response))

            return {
                'ttft': first_token_time - start_time if first_token_time else None,
                'topt': (last_token_time - first_token_time) / output_tokens if output_tokens > 0 else None,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'start': start_time,
                'end': end_time
            }

        except Exception as e:
            print(f"Request error: {str(e)}")
            return None

    def _generate_messages(self):
        """生成符合ChatML格式的messages结构"""
        char_count = random.randint(1200, 1300)
        user_content = '模型性能测试' * (char_count // 4)

        return [
            {"role": "user", "content": user_content},
            # 如果需要多轮对话可以添加更多message
            # {"role": "assistant", "content": "..."},
        ]

    async def _worker(self):
        """单个压测worker"""
        async with self.semaphore:
            messages = self._generate_messages()
            result = await self._async_stream_request(messages)
            if result and result['ttft'] is not None:
                self.results.append(result)

    async def run(self):
        """执行压测"""
        start_time = time.monotonic()

        tasks = [self._worker() for _ in range(self.num_requests)]
        await asyncio.gather(*tasks)

        self.total_duration = time.monotonic() - start_time

    def print_report(self):
        """生成测试报告"""
        if not self.results:
            print("No successful requests")
            return

        ttft = [r['ttft'] for r in self.results]
        topt = [r['topt'] for r in self.results if r['topt']]
        input_tokens = sum(r['input_tokens'] for r in self.results)
        output_tokens = sum(r['output_tokens'] for r in self.results)

        print("\n=== 异步压测结果 ===")
        print(f"并发数: {self.concurrency}")
        print(f"成功请求: {len(self.results)}/{self.num_requests}")
        print(f"总耗时: {self.total_duration:.2f}s")

        print("\nTTFT指标:")
        print(f"• 平均: {statistics.mean(ttft) * 1000:.2f}ms")
        if len(ttft) > 1:
            print(f"• 标准差: {statistics.stdev(ttft) * 1000:.2f}ms")
            print(f"• P95: {statistics.quantiles(ttft, n=100)[94] * 1000:.2f}ms")

        if topt:
            print("\nTOPT指标:")
            print(f"• 平均: {statistics.mean(topt) * 1000:.2f}ms/token")
            if len(topt) > 1:
                print(f"• 标准差: {statistics.stdev(topt) * 1000:.2f}ms")
                print(f"• P95: {statistics.quantiles(topt, n=100)[94] * 1000:.2f}ms")

        print("\n吞吐量:")
        print(f"输入 Tokens/s: {input_tokens / self.total_duration:.2f}")
        print(f"输出 Tokens/s: {output_tokens / self.total_duration:.2f}")
        print(f"RPM: {len(self.results) / self.total_duration * 60:.2f}")


async def main():
    # 配置参数
    config = {
        'base_url': "http://XXX.XXX.XXX.XXX:8000/v1",
        'model': "Qwen/QwQ-32B",
        'num_requests': 100,
        'concurrency': 20
    }

    print(f"启动异步压测 (请求数: {config['num_requests']}, 并发数: {config['concurrency']})")
    benchmark = AsyncLLMBenchmark(
        base_url=config['base_url'],
        model=config['model'],
        num_requests=config['num_requests'],
        concurrency=config['concurrency']
    )

    await benchmark.run()
    benchmark.print_report()


if __name__ == "__main__":
    asyncio.run(main())
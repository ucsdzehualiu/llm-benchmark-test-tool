import argparse
import asyncio
import random
import statistics
import time

from openai import AsyncOpenAI


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
        start_time = time.monotonic()
        first_token_time = None
        last_token_time = start_time
        full_response = []
        output_tokens = 0
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                # stop=["STOP_SEQUENCE_THAT_WILL_NEVER_APPEAR"]

            )

            async for chunk in stream:
                current_time = time.monotonic()
                delta_content = chunk.choices[0].delta.content or ""

                if delta_content and first_token_time is None:
                    first_token_time = current_time

                if delta_content:
                    last_token_time = current_time
                    full_response.append(delta_content)
                    output_tokens += 1
            end_time = time.monotonic()

            input_tokens = sum(len(msg["content"]) for msg in messages)
            output_duration = last_token_time - first_token_time if first_token_time else 0
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
        char_count = random.randint(self.min_chars, self.max_chars)
        # 更准确的中文token估算（按字计算）
        user_content = ''.join(random.choice('模型性能测试验证') for _ in range(char_count))
        return [{"role": "user", "content": user_content}]

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

import asyncio
from tabulate import tabulate
import matplotlib.pyplot as plt
from llm_benchmark import AsyncLLMBenchmark


async def run_concurrency_test(base_url, model, num_requests, concurrency,
                               max_tokens=512, temperature=0.7, min_chars=1200, max_chars=1300):
    benchmark = AsyncLLMBenchmark(
        base_url=base_url,
        model=model,
        num_requests=num_requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        min_chars=min_chars,
        max_chars=max_chars
    )
    await benchmark.run()
    return benchmark.generate_report()


async def main():
    base_url = "http://XX.XX.XX.XX:8000/v1"
    model = "Qwen/QwQ-32B"
    num_requests = 100
    concurrency_levels = [10,20,30,40,50]
    max_tokens = 700
    temperature = 0.7
    min_input_chars = 1200
    max_input_chars = 1300
    reports = []
    for concurrency in concurrency_levels:
        print(f"\nTesting with concurrency {concurrency}...")
        report = await run_concurrency_test(base_url, model, num_requests, concurrency,max_tokens,temperature, min_input_chars,max_input_chars)
        if report:
            reports.append(report)
        await asyncio.sleep(1)  # Brief pause to avoid server overload

    headers = [
        'Concurrency',
        'Successful Requests',
        'Total Duration (s)',
        'TTFT Mean (ms)',
        'TTFT P95 (ms)',
        'TPOT Mean (ms/token)',
        'Input Tokens/s',
        'Output Tokens/s (Total)',
        'Output Tokens/s (Avg)',
        'RPM'
    ]

    table_data = []
    for report in reports:
        # 添加调试打印
        print(f"\nDebug Report for concurrency {report['concurrency']}:")
        print(f"Output tokens avg: {report.get('output_tokens_per_sec_avg', 'missing')}")
        print(f"TPOT mean: {report.get('tpot_mean', 'missing')}")

        row = [
            report['concurrency'],
            f"{report['success']}/{report['total_requests']}",
            f"{report['total_duration']:.2f}",
            f"{report['ttft_mean'] * 1000:.2f}",
            f"{report['ttft_p95'] * 1000:.2f}" if report['success'] > 1 else 'N/A',
            f"{report['tpot_mean'] * 1000:.2f}" if report['tpot_mean'] else 'N/A',
            f"{report['input_tokens_per_sec']:.2f}",
            f"{report['output_tokens_per_sec_total']:.2f}",
            # 确保单位转换正确：1秒/tpot_mean = tokens/s
            f"{1 / report['tpot_mean']:.2f}" if report['tpot_mean'] else 'N/A',  # 修正计算
            f"{report['rpm']:.2f}"
        ]
        table_data.append(row)
    table_str = tabulate(table_data, headers=headers, tablefmt='fancy_grid')
    print("\n=== Benchmark Results Summary ===")
    print(table_str)

    # Save the table to a file
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write(table_str)
    concurrencies = [report['concurrency'] for report in reports]
    ttft_means = [report['ttft_mean'] * 1000 for report in reports]
    tpot_means = [report['tpot_mean'] * 1000 if report['tpot_mean'] else 0 for report in reports]
    output_tokens_sec_avg = [1 / report['tpot_mean'] if report['tpot_mean'] else 0 for report in reports]  # 直接计算
    rpms = [report['rpm'] for report in reports]

    # 创建2x2子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    # TTFT Mean
    ax1.plot(concurrencies, ttft_means, marker='o', linestyle='-', color='blue')
    ax1.set_title('TTFT Mean vs Concurrency')
    ax1.set_xlabel('Concurrency Level')
    ax1.set_ylabel('TTFT Mean (ms)')
    ax1.grid(True)

    # TPOT Mean
    ax2.plot(concurrencies, tpot_means, marker='o', linestyle='-', color='red')
    ax2.set_title('TPOT Mean vs Concurrency')
    ax2.set_xlabel('Concurrency Level')
    ax2.set_ylabel('TPOT Mean (ms/token)')
    ax2.grid(True)

    # RPM
    ax3.plot(concurrencies, rpms, marker='o', linestyle='-', color='green')
    ax3.set_title('RPM vs Concurrency')
    ax3.set_xlabel('Concurrency Level')
    ax3.set_ylabel('Requests Per Minute')
    ax3.grid(True)

    # Output Tokens/s (Avg)
    ax4.plot(concurrencies, output_tokens_sec_avg, marker='o', linestyle='-', color='purple')
    ax4.set_title('Output Tokens/s (Avg) vs Concurrency')
    ax4.set_xlabel('Concurrency Level')
    ax4.set_ylabel('Tokens per Second')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_chart.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())

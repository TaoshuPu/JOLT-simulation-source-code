from pathlib import Path

import jolt_hisc_ma_tool_mip_single_population_L60 as experiment


experiment.LLMS = 100
experiment.TOOLS = 300
experiment.SEED = 20260528 + 100 * 1000
experiment.OUT_DIR = Path("jolt_hisc_ma_tool_mip_single_population_L100_10min")
experiment.POP_SIZE = 256


if __name__ == "__main__":
    experiment.main()

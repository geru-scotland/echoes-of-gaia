from(bucket: "echoes_of_gaia")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "biome_states_15")
  |> keep(columns: ["_time", "_field", "_value"])
  |> pivot(
      rowKey: ["_time"],
      columnKey: ["_field"],
      valueColumn: "_value"
  )
  |> keep(columns: ["_time", "num_flora", "num_fauna", "avg_toxicity", "avg_size", "avg_growth_stage"])
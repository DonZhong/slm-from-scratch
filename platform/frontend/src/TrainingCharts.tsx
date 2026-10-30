import type { ReactNode } from "react";
import { Activity } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { buildSeries } from "./training-data";
import { number } from "./training-format";

const chartStyle = {
  background: "#10202c",
  border: "1px solid #304552",
  borderRadius: 8,
  color: "#edf4f5",
  fontSize: 12,
};
export function Stat({
  label,
  value,
  detail,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  detail: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className={`lab-stat ${accent ? "accent" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

export function Plot({
  title,
  subtitle,
  data,
  lines,
  selected,
  compact = false,
}: {
  title: string;
  subtitle: string;
  data: ReturnType<typeof buildSeries>;
  lines: { key: string; name: string; color: string; dashed?: boolean }[];
  selected: number | null;
  compact?: boolean;
}) {
  return (
    <section className={`lab-chart ${compact ? "compact" : ""}`}>
      <div className="lab-card-heading">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <Activity size={17} />
      </div>
      <div className="lab-chart-canvas">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 14, right: 15, left: 2, bottom: 8 }}
          >
            <CartesianGrid
              stroke="#243642"
              strokeDasharray="3 5"
              vertical={false}
            />
            <XAxis
              dataKey="global_step"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fill: "#8ba0ac", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              minTickGap={32}
            />
            <YAxis
              width={54}
              tick={{ fill: "#8ba0ac", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) =>
                Math.abs(value) > 0 && Math.abs(value) < 0.01
                  ? value.toExponential(0)
                  : number(value, 2)
              }
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={chartStyle}
              labelFormatter={(label) => `Step ${label}`}
              formatter={(value, name) => [number(value, 6), name]}
            />
            <Legend
              verticalAlign="top"
              align="right"
              iconType="plainline"
              wrapperStyle={{ fontSize: 11, paddingBottom: 15 }}
            />
            {selected !== null && (
              <ReferenceLine
                x={selected}
                stroke="#deeaf2"
                strokeDasharray="3 3"
              />
            )}
            {lines.map((line) => (
              <Line
                key={line.key}
                type="linear"
                dataKey={line.key}
                name={line.name}
                stroke={line.color}
                strokeWidth={line.dashed ? 1.5 : 2}
                strokeDasharray={line.dashed ? "4 4" : undefined}
                dot={
                  line.key === "val_loss" || data.length < 3 ? { r: 3 } : false
                }
                activeDot={{ r: 4 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="lab-chart-axis">优化器更新步数 · GLOBAL STEP</div>
    </section>
  );
}

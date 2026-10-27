
import TrainingPanel from "./TrainingPanel";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ComponentType, CSSProperties } from "react";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Layers,
  MemoryStick,
  Monitor,
  Network,
  Radio,
  RefreshCw,
  Server,
  ShieldCheck,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

type SectionKey =
  | "gpu"
  | "cpu"
  | "memory"
  | "disk"
  | "network"
  | "system";

type Metrics = Record<string, any>;

type Snapshot = {
  id: number;
  sampled_at: string;
  received_at?: string;
  host_name: string;
  training_run_id: number | null;
  metrics: Metrics;
};


type ChartPoint = {
  time: string;
  timestamp: number;

  // 资源总览已有的指标
  gpu: number | null;
  cpu: number | null;
  memory: number | null;
  disk: number | null;

  // GPU 深度监控新增指标
  gpuMemoryMB: number | null;
  gpuTemperatureC: number | null;
  gpuPowerW: number | null;

  // 磁盘吞吐：字节/秒
  diskReadBps: number | null;
  diskWriteBps: number | null;

  // 网络吞吐：字节/秒
  netRecvBps: number | null;
  netSentBps: number | null;

};


type SectionMeta = {
  title: string;
  code: string;
  description: string;
  icon: ComponentType<{ size?: number; className?: string }>;
  accent: string;
};

const sectionMeta: Record<SectionKey, SectionMeta> = {
  gpu: {
    title: "GPU 加速器",
    code: "GPU / COMPUTE",
    description: "显卡计算、显存、温度、功耗与时钟",
    icon: Gauge,
    accent: "#54E7FF",
  },
  cpu: {
    title: "CPU 处理器",
    code: "CPU / PROCESSOR",
    description: "处理器利用率、逐核心负载与系统负载",
    icon: Cpu,
    accent: "#A69AFF",
  },
  memory: {
    title: "系统内存",
    code: "MEMORY / RAM",
    description: "内存使用、可用空间、交换分区",
    icon: MemoryStick,
    accent: "#48E3B5",
  },
  disk: {
    title: "磁盘存储",
    code: "STORAGE / DISK",
    description: "挂载点、容量、可用空间与磁盘 I/O",
    icon: HardDrive,
    accent: "#FFBD6B",
  },
  network: {
    title: "网络接口",
    code: "NETWORK / I/O",
    description: "网卡、收发字节、数据包与错误计数",
    icon: Network,
    accent: "#64A9FF",
  },
  system: {
    title: "系统信息",
    code: "HOST / RUNTIME",
    description: "主机、操作系统、Python 与采集器版本",
    icon: Server,
    accent: "#F28EFF",
  },
};

const numberOf = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value)
    ? value
    : null;

const objectOf = (
  value: unknown,
): Metrics => {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Metrics;
  }
  return {};
};

const arrayOf = (value: unknown): Metrics[] =>
  Array.isArray(value) ? value : [];

const getObject = (
  snapshot: Snapshot | null,
  key: string,
): Metrics => objectOf(snapshot?.metrics?.[key]);

const formatNumber = (
  value: unknown,
  digits = 1,
): string => {
  const n = numberOf(value);
  if (n === null) return "—";

  return n.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const formatInteger = (value: unknown): string =>
  formatNumber(value, 0);

const formatPercent = (value: unknown): string => {
  const n = numberOf(value);
  return n === null ? "—" : `${formatNumber(n)}%`;
};

const formatMB = (value: unknown): string => {
  const n = numberOf(value);
  return n === null ? "—" : `${formatNumber(n)} MB`;
};

const formatTemperature = (value: unknown): string => {
  const n = numberOf(value);
  return n === null ? "—" : `${formatNumber(n, 0)} °C`;
};

const formatBytes = (value: unknown): string => {
  let n = numberOf(value);
  if (n === null) return "—";

  const units = ["B", "KB", "MB", "GB", "TB"];
  let unit = 0;

  while (n >= 1024 && unit < units.length - 1) {
    n /= 1024;
    unit += 1;
  }

  return `${formatNumber(n)} ${units[unit]}`;
};

const formatRate = (value: unknown): string => {
  const n = numberOf(value);
  return n === null ? "等待采样" : `${formatBytes(n)}/s`;
};

const formatDateTime = (value?: string): string => {
  if (!value) return "暂无数据";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString("zh-CN", {
    hour12: false,
  });
};

const formatClock = (value?: string): string => {
  if (!value) return "—";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";

  return date.toLocaleTimeString("zh-CN", {
    hour12: false,
  });
};

const clampPercent = (value: unknown): number => {
  const n = numberOf(value) ?? 0;
  return Math.max(0, Math.min(100, n));
};

const getDiskMount = (
  snapshot: Snapshot | null,
): Metrics | null => {
  const mounts = arrayOf(getObject(snapshot, "disk").mounts);

  return (
    mounts.find((mount) => mount.path === "/mnt/d") ??
    mounts[0] ??
    null
  );
};

const getHealth = (
  snapshot: Snapshot | null,
): "waiting" | "nominal" | "watch" | "critical" => {
  if (!snapshot) return "waiting";

  const gpu = getObject(snapshot, "gpu");
  const cpu = getObject(snapshot, "cpu");
  const memory = getObject(snapshot, "memory");
  const disk = getDiskMount(snapshot);

  const gpuTemperature = numberOf(gpu.temperature_c);
  const gpuMemoryPercent =
    numberOf(gpu.memory_total_mb) &&
    numberOf(gpu.memory_used_mb) !== null
      ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100
      : null;

  const cpuPercent = numberOf(cpu.utilization_percent);
  const memoryPercent = numberOf(memory.percent);
  const diskPercent = numberOf(disk?.percent);

  if (
    (gpuTemperature !== null && gpuTemperature >= 85) ||
    (gpuMemoryPercent !== null && gpuMemoryPercent >= 95) ||
    (cpuPercent !== null && cpuPercent >= 95) ||
    (memoryPercent !== null && memoryPercent >= 95) ||
    (diskPercent !== null && diskPercent >= 95)
  ) {
    return "critical";
  }

  if (
    (gpuTemperature !== null && gpuTemperature >= 75) ||
    (gpuMemoryPercent !== null && gpuMemoryPercent >= 85) ||
    (cpuPercent !== null && cpuPercent >= 85) ||
    (memoryPercent !== null && memoryPercent >= 85) ||
    (diskPercent !== null && diskPercent >= 85)
  ) {
    return "watch";
  }

  return "nominal";
};

function MetricCard({
  section,
  snapshot,
  onOpen,
}: {
  section: SectionKey;
  snapshot: Snapshot | null;
  onOpen: (section: SectionKey) => void;
}) {
  const meta = sectionMeta[section];
  const Icon = meta.icon;
  const gpu = getObject(snapshot, "gpu");
  const cpu = getObject(snapshot, "cpu");
  const memory = getObject(snapshot, "memory");
  const disk = getDiskMount(snapshot);
  const network = getObject(snapshot, "network");
  const system = getObject(snapshot, "system");

  let value = "—";
  let sub = "等待数据";
  let progress: number | null = null;
  let badge = "NO DATA";

  switch (section) {
    case "gpu":
      value =
        gpu.available === false
          ? "N/A"
          : formatPercent(gpu.utilization_percent);
      sub = `${formatMB(gpu.memory_used_mb)} / ${formatMB(gpu.memory_total_mb)} 显存`;
      progress = numberOf(gpu.utilization_percent);
      badge = gpu.available === false ? "UNAVAILABLE" : "COMPUTE";
      break;

    case "cpu":
      value = formatPercent(cpu.utilization_percent);
      sub = `${formatInteger(cpu.physical_cores)} 物理核心 · ${formatInteger(cpu.logical_cores)} 逻辑核心`;
      progress = numberOf(cpu.utilization_percent);
      badge = "PROCESSOR";
      break;

    case "memory":
      value = formatPercent(memory.percent);
      sub = `${formatMB(memory.used_mb)} / ${formatMB(memory.total_mb)}`;
      progress = numberOf(memory.percent);
      badge = "RAM";
      break;

    case "disk":
      value = formatPercent(disk?.percent);
      sub = `${disk?.path ?? "未知挂载点"} · ${formatNumber(disk?.free_gb)} GB 可用`;
      progress = numberOf(disk?.percent);
      badge = "STORAGE";
      break;

    case "network": {
      const interfaces = arrayOf(network.interfaces);
      value = `${interfaces.length} 个接口`;
      sub = `接收 ${formatRate(network.bytes_recv_per_second)} · 发送 ${formatRate(network.bytes_sent_per_second)}`;
      badge = "NETWORK I/O";
      break;
    }

    case "system":
      value = snapshot ? "ONLINE" : "OFFLINE";
      sub = String(system.hostname ?? snapshot?.host_name ?? "等待主机信息");
      badge = "HOST";
      break;
  }

  return (
    <button
      className="metric-card"
      type="button"
      onClick={() => onOpen(section)}
      style={{ "--accent": meta.accent } as CSSProperties}
    >
      <div className="metric-card-top">
        <div className="metric-icon">
          <Icon size={21} />
        </div>
        <span className="metric-code">{badge}</span>
        <ChevronRight className="metric-chevron" size={17} />
      </div>

      <div className="metric-card-title">{meta.title}</div>
      <div className="metric-card-value">{value}</div>
      <div className="metric-card-sub">{sub}</div>

      {progress !== null && (
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${clampPercent(progress)}%` }}
          />
        </div>
      )}

      <div className="metric-card-footer">
        <span>{meta.code}</span>
        <span>查看详情 ↗</span>
      </div>
    </button>
  );
}

function MetricTile({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note?: string;
}) {
  return (
    <div className="detail-tile">
      <div className="detail-tile-label">{label}</div>
      <div className="detail-tile-value">{value}</div>
      {note && <div className="detail-tile-note">{note}</div>}
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="empty-state">
      <Activity size={23} />
      <span>{text}</span>
    </div>
  );
}


function GpuHistoryPanel({
  points,
}: {
  points: ChartPoint[];
}) {
  const charts = [
    {
      key: "gpu",
      title: "GPU 利用率",
      unit: "%",
      color: "#54E7FF",
      domain: [0, 100] as [number, number],
    },
    {
      key: "gpuMemoryMB",
      title: "显存占用",
      unit: "MB",
      color: "#A69AFF",
      domain: [0, "auto"] as [number, string],
    },
    {
      key: "gpuTemperatureC",
      title: "GPU 温度",
      unit: "°C",
      color: "#FFBD6B",
      domain: [0, "auto"] as [number, string],
    },
    {
      key: "gpuPowerW",
      title: "GPU 功耗",
      unit: "W",
      color: "#48E3B5",
      domain: [0, "auto"] as [number, string],
    },
  ] as const;

  return (
    <section className="gpu-history-section">
      <h3 className="detail-subtitle">
        GPU 历史遥测 / GPU TELEMETRY
      </h3>

      <div className="gpu-history-grid">
        {charts.map((chart) => {
          const validCount = points.filter(
            (point) => point[chart.key] !== null,
          ).length;

          return (
            <div className="gpu-history-card" key={chart.key}>
              <div className="gpu-history-heading">
                <span>{chart.title}</span>
                <span>{chart.unit}</span>
              </div>

              {validCount < 2 ? (
                <EmptyState
                  text="有效采样点不足，等待新的 GPU 数据。"
                />
              ) : (
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart
                    data={points}
                    margin={{
                      top: 12,
                      right: 8,
                      bottom: 0,
                      left: -14,
                    }}
                  >
                    <CartesianGrid
                      stroke="#1A3048"
                      strokeDasharray="4 4"
                      vertical={false}
                    />

                    <XAxis
                      dataKey="time"
                      tick={{
                        fill: "#8199B5",
                        fontSize: 9,
                      }}
                      axisLine={false}
                      tickLine={false}
                      minTickGap={22}
                    />

                    <YAxis
                      domain={chart.domain}
                      tick={{
                        fill: "#8199B5",
                        fontSize: 9,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <Tooltip
                      contentStyle={{
                        background: "#091525",
                        border: "1px solid #29425D",
                        borderRadius: 10,
                      }}
                    />

                    <Line
                      type="monotone"
                      dataKey={chart.key}
                      name={`${chart.title} (${chart.unit})`}
                      stroke={chart.color}
                      strokeWidth={2}
                      dot={false}
                      connectNulls={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}


type ThroughputKind = "disk" | "network";

type ThroughputKey =
  | "diskReadBps"
  | "diskWriteBps"
  | "netRecvBps"
  | "netSentBps";

type ThroughputSeries = {
  key: ThroughputKey;
  name: string;
  color: string;
};

const throughputConfigs: Record<
  ThroughputKind,
  {
    title: string;
    description: string;
    series: ThroughputSeries[];
  }
> = {
  disk: {
    title: "磁盘实时吞吐",
    description: "DISK READ / WRITE · BYTES PER SECOND",
    series: [
      {
        key: "diskReadBps",
        name: "磁盘读取",
        color: "#FFBD6B",
      },
      {
        key: "diskWriteBps",
        name: "磁盘写入",
        color: "#54E7FF",
      },
    ],
  },
  network: {
    title: "网络实时吞吐",
    description: "NETWORK RECEIVE / SEND · BYTES PER SECOND",
    series: [
      {
        key: "netRecvBps",
        name: "网络接收",
        color: "#54E7FF",
      },
      {
        key: "netSentBps",
        name: "网络发送",
        color: "#48E3B5",
      },
    ],
  },
};

function ThroughputHistoryPanel({
  kind,
  points,
}: {
  kind: ThroughputKind;
  points: ChartPoint[];
}) {
  const config = throughputConfigs[kind];

  const validPoints = points.filter((point) =>
    config.series.some(
      (series) => point[series.key] !== null,
    ),
  );

  return (
    <section className="throughput-section">
      <h3 className="detail-subtitle">
        {config.title}
      </h3>

      <div className="throughput-panel">
        <div className="throughput-heading">
          <span>{config.description}</span>
          <span>{validPoints.length} 个有效采样点</span>
        </div>

        {validPoints.length < 2 ? (
          <EmptyState
            text="有效吞吐采样不足。请保持采集器持续运行。"
          />
        ) : (
          <div className="throughput-chart">
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <LineChart
                data={points}
                margin={{
                  top: 12,
                  right: 15,
                  bottom: 0,
                  left: 2,
                }}
              >
                <CartesianGrid
                  stroke="#1A3048"
                  strokeDasharray="4 4"
                  vertical={false}
                />

                <XAxis
                  dataKey="time"
                  tick={{
                    fill: "#8199B5",
                    fontSize: 10,
                  }}
                  axisLine={false}
                  tickLine={false}
                  minTickGap={24}
                />

                <YAxis
                  width={68}
                  domain={[0, "auto"]}
                  tickFormatter={(value: number) =>
                    formatBytes(value)
                  }
                  tick={{
                    fill: "#8199B5",
                    fontSize: 10,
                  }}
                  axisLine={false}
                  tickLine={false}
                />

                <Tooltip
                  formatter={(value) => formatRate(value)}
                  contentStyle={{
                    background: "#091525",
                    border: "1px solid #29425D",
                    borderRadius: 10,
                  }}
                />

                {config.series.map((series) => (
                  <Line
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    name={series.name}
                    stroke={series.color}
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="throughput-legend">
          {config.series.map((series) => {
            const current =
              points.length > 0
                ? points[points.length - 1][series.key]
                : null;

            return (
              <div
                className="throughput-legend-item"
                key={series.key}
              >
                <span
                  className="throughput-legend-dot"
                  style={{
                    background: series.color,
                  }}
                />
                <span>{series.name}</span>
                <strong>{formatRate(current)}</strong>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}


function DetailContent({
  section,
  snapshot,
}: {
  section: SectionKey;
  snapshot: Snapshot | null;
}) {
  const data = getObject(snapshot, section);

  if (!snapshot) {
    return <EmptyState text="尚未收到硬件快照，请检查采集器是否运行。" />;
  }

  switch (section) {
    case "gpu": {
      const devices = arrayOf(data.devices);

      return (
        <>
          <div className="detail-grid">
            <MetricTile label="GPU 利用率" value={formatPercent(data.utilization_percent)} />
            <MetricTile label="显存已用" value={formatMB(data.memory_used_mb)} />
            <MetricTile label="显存总量" value={formatMB(data.memory_total_mb)} />
            <MetricTile label="GPU 温度" value={formatTemperature(data.temperature_c)} />
            <MetricTile label="当前功耗" value={numberOf(data.power_draw_w) === null ? "—" : `${formatNumber(data.power_draw_w)} W`} />
            <MetricTile label="功耗上限" value={numberOf(data.power_limit_w) === null ? "暂不可获取" : `${formatNumber(data.power_limit_w)} W`} />
            <MetricTile label="图形时钟" value={numberOf(data.graphics_clock_mhz) === null ? "—" : `${formatInteger(data.graphics_clock_mhz)} MHz`} />
            <MetricTile label="显存时钟" value={numberOf(data.memory_clock_mhz) === null ? "—" : `${formatInteger(data.memory_clock_mhz)} MHz`} />
          </div>

          <h3 className="detail-subtitle">GPU 设备清单</h3>

          {devices.length === 0 ? (
            <EmptyState text="没有单独的设备明细。" />
          ) : (
            devices.map((device, index) => (
              <div className="detail-subcard" key={`${device.name ?? "gpu"}-${index}`}>
                <div className="detail-subcard-heading">
                  <Gauge size={18} />
                  <strong>{device.name ?? `GPU ${index}`}</strong>
                </div>
                <div className="detail-inline-grid">
                  <span>利用率：{formatPercent(device.utilization_percent)}</span>
                  <span>温度：{formatTemperature(device.temperature_c)}</span>
                  <span>显存：{formatMB(device.memory_used_mb)} / {formatMB(device.memory_total_mb)}</span>
                  <span>功耗：{numberOf(device.power_draw_w) === null ? "—" : `${formatNumber(device.power_draw_w)} W`}</span>
                </div>
              </div>
            ))
          )}
        </>
      );
    }

    case "cpu": {

      const cores: (number | null)[] =
        Array.isArray(data.per_core_utilization_percent)
          ? data.per_core_utilization_percent.map(
              (value: unknown) => numberOf(value),
            )
          : [];

      const load = objectOf(data.load_average);
      const temperatures = arrayOf(data.temperature_sensors);

      return (
        <>
          <div className="detail-grid">
            <MetricTile label="总体利用率" value={formatPercent(data.utilization_percent)} />
            <MetricTile label="物理核心" value={formatInteger(data.physical_cores)} />
            <MetricTile label="逻辑核心" value={formatInteger(data.logical_cores)} />
            <MetricTile label="系统负载 · 1 分钟" value={formatNumber(load["1m"], 2)} />
            <MetricTile label="系统负载 · 5 分钟" value={formatNumber(load["5m"], 2)} />
            <MetricTile label="系统负载 · 15 分钟" value={formatNumber(load["15m"], 2)} />
          </div>

          <h3 className="detail-subtitle">逐逻辑核心利用率</h3>


          {cores.length === 0 ? (
            <EmptyState text="当前没有逐核心数据。" />
          ) : (
            <div className="cpu-core-grid">
              {cores.map((load, index) => {
                const level =
                  load === null
                    ? "unknown"
                    : load >= 85
                      ? "critical"
                      : load >= 60
                        ? "high"
                        : load >= 25
                          ? "medium"
                          : "low";

                return (
                  <div
                    className={`cpu-core-cell ${level}`}
                    key={index}
                    title={`逻辑核心 ${index}：${formatPercent(load)}`}
                  >
                    <div className="cpu-core-header">
                      <span>
                        CORE {String(index).padStart(2, "0")}
                      </span>
                      <Activity size={14} />
                    </div>

                    <strong>{formatPercent(load)}</strong>

                    <div className="cpu-core-track">
                      <div
                        style={{
                          width: `${clampPercent(load)}%`,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}


          <h3 className="detail-subtitle">温度传感器</h3>

          {temperatures.length === 0 ? (
            <EmptyState text="当前环境未返回 CPU 温度传感器数据。这不代表 CPU 没有温度。" />
          ) : (
            <div className="detail-inline-grid">
              {temperatures.map((sensor, index) => (
                <span key={index}>
                  {sensor.label ?? sensor.name ?? `传感器 ${index + 1}`}：
                  {formatTemperature(sensor.current)}
                </span>
              ))}
            </div>
          )}
        </>
      );
    }

    case "memory": {
      const swap = objectOf(data.swap);

      return (
        <div className="detail-grid">
          <MetricTile label="内存利用率" value={formatPercent(data.percent)} />
          <MetricTile label="已用内存" value={formatMB(data.used_mb)} />
          <MetricTile label="可用内存" value={formatMB(data.available_mb)} />
          <MetricTile label="内存总量" value={formatMB(data.total_mb)} />
          <MetricTile label="Swap 利用率" value={formatPercent(swap.percent)} />
          <MetricTile label="Swap 已用" value={formatMB(swap.used_mb)} />
          <MetricTile label="Swap 总量" value={formatMB(swap.total_mb)} />
        </div>
      );
    }

    case "disk": {
      const mounts = arrayOf(data.mounts);
      const io = objectOf(data.io);

      return (
        <>
          <div className="detail-grid">
            <MetricTile label="累计读取" value={formatBytes(io.read_bytes_total)} />
            <MetricTile label="累计写入" value={formatBytes(io.write_bytes_total)} />
            <MetricTile label="读取速度" value={formatRate(io.read_bytes_per_second)} />
            <MetricTile label="写入速度" value={formatRate(io.write_bytes_per_second)} />
          </div>

          <h3 className="detail-subtitle">挂载点与容量</h3>

          {mounts.length === 0 ? (
            <EmptyState text="未获取到磁盘挂载点。" />
          ) : (
            mounts.map((mount) => (
              <div className="detail-subcard" key={mount.path}>
                <div className="detail-subcard-heading">
                  <HardDrive size={18} />
                  <strong>{mount.path}</strong>
                  <span className="detail-subcard-right">{formatPercent(mount.percent)}</span>
                </div>
                <div className="progress-track mount-progress">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${clampPercent(mount.percent)}%`,
                      background: "#FFBD6B",
                    }}
                  />
                </div>
                <div className="detail-inline-grid">
                  <span>总容量：{formatNumber(mount.total_gb)} GB</span>
                  <span>已用：{formatNumber(mount.used_gb)} GB</span>
                  <span>可用：{formatNumber(mount.free_gb)} GB</span>
                </div>
              </div>
            ))
          )}
        </>
      );
    }

    case "network": {
      const interfaces = arrayOf(data.interfaces);

      return (
        <>
          <div className="detail-grid">
            <MetricTile label="累计接收" value={formatBytes(data.bytes_recv_total)} />
            <MetricTile label="累计发送" value={formatBytes(data.bytes_sent_total)} />
            <MetricTile label="接收速度" value={formatRate(data.bytes_recv_per_second)} />
            <MetricTile label="发送速度" value={formatRate(data.bytes_sent_per_second)} />
            <MetricTile label="接口数量" value={formatInteger(interfaces.length)} />
          </div>

          <h3 className="detail-subtitle">接口明细</h3>

          {interfaces.length === 0 ? (
            <EmptyState text="没有网络接口数据。" />
          ) : (
            interfaces.map((item) => (
              <div className="detail-subcard" key={item.name}>
                <div className="detail-subcard-heading">
                  <Network size={18} />
                  <strong>{item.name}</strong>
                </div>
                <div className="detail-inline-grid">
                  <span>接收：{formatBytes(item.bytes_recv_total)}</span>
                  <span>发送：{formatBytes(item.bytes_sent_total)}</span>
                  <span>接收数据包：{formatInteger(item.packets_recv_total)}</span>
                  <span>发送数据包：{formatInteger(item.packets_sent_total)}</span>
                  <span>接收错误：{formatInteger(item.errors_in)}</span>
                  <span>发送错误：{formatInteger(item.errors_out)}</span>
                  <span>接收丢包：{formatInteger(item.drops_in)}</span>
                  <span>发送丢包：{formatInteger(item.drops_out)}</span>
                </div>
              </div>
            ))
          )}
        </>
      );
    }

    case "system":
      return (
        <div className="detail-grid">
          <MetricTile label="主机名称" value={String(data.hostname ?? snapshot.host_name)} />
          <MetricTile label="操作系统" value={String(data.platform ?? "—")} />
          <MetricTile label="处理器架构" value={String(data.machine ?? "—")} />
          <MetricTile label="Python 版本" value={String(data.python_version ?? "—")} />
          <MetricTile label="采集器版本" value={String(data.collector_version ?? "—")} />
          <MetricTile label="快照编号" value={String(snapshot.id)} />
          <MetricTile label="采样时间" value={formatDateTime(snapshot.sampled_at)} />
          <MetricTile label="接收时间" value={formatDateTime(snapshot.received_at)} />
          <MetricTile label="训练实验 ID" value={snapshot.training_run_id === null ? "未关联" : String(snapshot.training_run_id)} />
        </div>
      );
  }
}

function DetailModal({
  section,
  snapshot,
  chartData,
  onClose,
}: {
  section: SectionKey;
  snapshot: Snapshot | null;
  chartData: ChartPoint[];
  onClose: () => void;
}) {
  const meta = sectionMeta[section];
  const Icon = meta.icon;
  const trendKey: Partial<Record<SectionKey, keyof ChartPoint>> = {
    cpu: "cpu",
    memory: "memory",
    disk: "disk",
  };
  const dataKey = trendKey[section];
  const raw = getObject(snapshot, section);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="detail-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="detail-title"
        style={{ "--accent": meta.accent } as CSSProperties}
      >
        <header className="modal-header">
          <div className="modal-heading">
            <div className="metric-icon">
              <Icon size={22} />
            </div>
            <div>
              <div className="eyebrow">{meta.code}</div>
              <h2 id="detail-title">{meta.title}</h2>
              <p>{meta.description}</p>
            </div>
          </div>

          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="关闭详情"
          >
            <X size={20} />
          </button>
        </header>

        <div className="modal-content">
          <div className="detail-section-heading">
            <span>指标详情 / METRIC INSPECTION</span>
            <span>SNAPSHOT #{snapshot?.id ?? "—"}</span>
          </div>

          <DetailContent section={section} snapshot={snapshot} />

          {(section === "disk" || section === "network") && (
            <ThroughputHistoryPanel
              kind={section}
              points={chartData}
            />
          )}


          {section === "gpu" && (
            <GpuHistoryPanel points={chartData} />
          )}

          {dataKey && (
            <>
              <h3 className="detail-subtitle">历史变化趋势</h3>
              <div className="modal-chart">
                {chartData.length < 2 ? (
                  <EmptyState text="至少需要两个有效采样点才能观察趋势。" />
                ) : (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={chartData}>
                      <CartesianGrid stroke="#1A3048" strokeDasharray="4 4" />
                      <XAxis
                        dataKey="time"
                        tick={{ fill: "#8199B5", fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 100]}
                        tick={{ fill: "#8199B5", fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#091525",
                          border: "1px solid #29425D",
                          borderRadius: 10,
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey={dataKey}
                        name={meta.title}
                        stroke={meta.accent}
                        strokeWidth={2}
                        dot={false}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </>
          )}

          <details className="raw-details">
            <summary>查看原始 JSON 数据</summary>
            <pre>{JSON.stringify(raw, null, 2)}</pre>
          </details>

          <div className="modal-footnote">
            <ShieldCheck size={15} />
            数据来源：硬件采集器 → C++ API → PostgreSQL
          </div>
        </div>
      </section>
    </div>
  );
}

export default function App() {
  const [latest, setLatest] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<Snapshot[]>([]);
  const [apiOnline, setApiOnline] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedSection, setSelectedSection] = useState<SectionKey | null>(null);
  const [clockNow, setClockNow] = useState<number | null>(null);
  const inFlight = useRef(false);

  const fetchSnapshots = useCallback(async (showRefreshing = true) => {
  if (inFlight.current) return;
  inFlight.current = true;

  if (showRefreshing) {
    setIsRefreshing(true);
  }

    try {
      const [latestResponse, historyResponse] = await Promise.all([
        fetch("/api/hardware-snapshots/latest", { cache: "no-store" }),
        fetch("/api/hardware-snapshots?limit=60", { cache: "no-store" }),
      ]);

      if (!latestResponse.ok || !historyResponse.ok) {
        throw new Error(
          `API HTTP ${latestResponse.status}/${historyResponse.status}`,
        );
      }

      const latestPayload = await latestResponse.json();
      const historyPayload = await historyResponse.json();

      if (
        latestPayload.status !== "ok" ||
        historyPayload.status !== "ok"
      ) {
        throw new Error("API 返回了异常状态");
      }

      setLatest(latestPayload.item ?? null);
      setHistory(
        Array.isArray(historyPayload.items)
          ? historyPayload.items
          : [],
      );
      setApiOnline(true);
      setError("");
      setLastUpdated(new Date());
    } catch (caught) {
      setApiOnline(false);
      setError(
        caught instanceof Error ? caught.message : "无法连接监控 API",
      );
      } finally {
        inFlight.current = false;

        if (showRefreshing) {
          setIsRefreshing(false);
        }
      }
  }, []);

  useEffect(() => {
    const initialTimer = window.setTimeout(() => {
      void fetchSnapshots(false);
    }, 0);

    const refreshTimer = window.setInterval(() => {
      void fetchSnapshots(false);
    }, 5000);

    return () => {
      window.clearTimeout(initialTimer);
      window.clearInterval(refreshTimer);
    };
  }, [fetchSnapshots]);

  useEffect(() => {
  const updateClock = () => setClockNow(Date.now());

  const initialTimer = window.setTimeout(updateClock, 0);
  const clockTimer = window.setInterval(updateClock, 1000);

  return () => {
    window.clearTimeout(initialTimer);
    window.clearInterval(clockTimer);
    };
  }, []);

  const chartData = useMemo<ChartPoint[]>(() => {
    if (!latest) return [];

    const matchingHost = history.filter(
      (item) => item.host_name === latest.host_name,
    );

    const source = matchingHost;
    const sorted = [...source].sort(
      (a, b) =>
        new Date(a.sampled_at).getTime() -
        new Date(b.sampled_at).getTime(),
    );

    if (!sorted.some((item) => item.id === latest.id)) {
      sorted.push(latest);
      sorted.sort(
        (a, b) =>
          new Date(a.sampled_at).getTime() -
          new Date(b.sampled_at).getTime(),
      );
    }

    return sorted.slice(-60).map((snapshot) => {
      const gpu = getObject(snapshot, "gpu");
      const cpu = getObject(snapshot, "cpu");
      const memory = getObject(snapshot, "memory");
      const disk = getDiskMount(snapshot);

      const diskIo = objectOf(
        getObject(snapshot, "disk").io,
      );

      const network = getObject(snapshot, "network");

      const timestamp = new Date(snapshot.sampled_at).getTime();


      return {
        time: formatClock(snapshot.sampled_at),
        timestamp,

        gpu: numberOf(gpu.utilization_percent),
        cpu: numberOf(cpu.utilization_percent),
        memory: numberOf(memory.percent),
        disk: numberOf(disk?.percent),

        gpuMemoryMB: numberOf(gpu.memory_used_mb),
        gpuTemperatureC: numberOf(gpu.temperature_c),
        gpuPowerW: numberOf(gpu.power_draw_w),

        diskReadBps: numberOf(
          diskIo.read_bytes_per_second,
        ),

        diskWriteBps: numberOf(
          diskIo.write_bytes_per_second,
        ),

        netRecvBps: numberOf(
          network.bytes_recv_per_second,
        ),

        netSentBps: numberOf(
          network.bytes_sent_per_second,
        ),

      };

    });
  }, [history, latest]);

  const ageSeconds =
    latest && clockNow !== null
      ? Math.max(
          0,
          Math.floor(
            (clockNow - new Date(latest.sampled_at).getTime()) / 1000,
          ),
        )
      : null;


  // 快照超过 20 秒未更新，判定为数据过期
  const snapshotFresh =
    ageSeconds !== null &&
    Number.isFinite(ageSeconds) &&
    ageSeconds <= 20;

  // 数据过期时不继续报告硬件“运行正常”
  const health = snapshotFresh
    ? getHealth(latest)
    : "waiting";

  const ageText =
    ageSeconds === null || !Number.isFinite(ageSeconds)
      ? "等待首个快照"
      : ageSeconds < 60
        ? `${ageSeconds} 秒前采样`
        : `${Math.floor(ageSeconds / 60)} 分钟前采样`;

  const gpu = getObject(latest, "gpu");
  const cpu = getObject(latest, "cpu");
  const memory = getObject(latest, "memory");
  const disk = getDiskMount(latest);

  const healthText = !latest
    ? "等待首个快照"
    : !snapshotFresh
      ? "采集数据已过期"
      : {
          waiting: "等待数据",
          nominal: "运行正常",
          watch: "需要关注",
          critical: "高负载预警",
        }[health];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview">
          <div className="brand-mark">
            <Layers size={23} />
          </div>
          <div className="brand-copy">
            <strong>SLM<span> / LAB</span></strong>
            <small>HARDWARE INTELLIGENCE</small>
          </div>
        </a>

        <div className="side-label">OBSERVABILITY</div>

        <a className="nav-item active" href="#overview">
          <Monitor size={18} />
          <span>监控总览</span>
          <span className="nav-index">01</span>
        </a>

        <a className="nav-item" href="#resource-trends">
          <Activity size={18} />
          <span>资源趋势</span>
          <span className="nav-index">02</span>
        </a>

        <a className="nav-item" href="#components">
          <Database size={18} />
          <span>硬件组件</span>
          <span className="nav-index">03</span>
        </a>

        <a
          className="nav-item"
          href="#training-monitor"
        >
          <BrainCircuit size={18} />
          <span>模型训练</span>
          <span className="nav-index">04</span>
        </a>

        <div className="sidebar-spacer" />

        <div className="sidebar-status">
          <div className="side-label">LOCAL ENVIRONMENT</div>
          <div className="sidebar-host">
            <span className={`status-dot ${apiOnline ? "online" : ""}`} />
            <span>{apiOnline ? "API CONNECTED" : "API DISCONNECTED"}</span>
          </div>
          <div className="sidebar-hostname">
            {latest?.host_name ?? "等待主机连接"}
          </div>
          <div className="sidebar-version">SLM TELEMETRY · V1.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            <span>SLM</span>
            <ChevronRight size={14} />
            <span>OBSERVABILITY</span>
            <ChevronRight size={14} />
            <strong>OVERVIEW</strong>
          </div>

          <div className="topbar-actions">
            <div className={`connection-pill ${apiOnline ? "connected" : ""}`}>
              <span className="status-dot" />
              {apiOnline ? "API ONLINE" : "API OFFLINE"}
            </div>

            <button
              className="refresh-button"
              type="button"
              onClick={() => void fetchSnapshots()}
              disabled={isRefreshing}
            >
              <RefreshCw
                size={16}
                className={isRefreshing ? "spin" : ""}
              />
              <span>刷新数据</span>
            </button>
          </div>
        </header>

        <section className="hero" id="overview">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              SLM-FROM-SCRATCH / HARDWARE TELEMETRY
            </div>

            <h1>
              硬件监控<span>指挥中心</span>
            </h1>

            <p className="hero-description">
              从 GPU 计算到系统资源，从单项指标到历史变化，
              全面观察你的模型训练环境。
            </p>

            <div className="hero-meta">
              <span><Server size={15} /> {latest?.host_name ?? "主机等待连接"}</span>
              <span><Radio size={15} /> 每 5 秒刷新</span>
              <span><Clock3 size={15} /> {ageText}</span>
            </div>

            {error && (
              <div className="error-banner">
                <AlertTriangle size={16} />
                <span>监控 API 连接失败：{error}</span>
              </div>
            )}
          </div>

          <div className="hero-orbit">
            <div className="orbit-ring orbit-ring-outer" />
            <div className="orbit-ring orbit-ring-inner" />
            <div className="orbit-core">
              <Zap size={32} />
            </div>
            <div className="orbit-caption">
              <span>SYSTEM STATUS</span>
              <strong>{healthText}</strong>
            </div>
          </div>
        </section>

        <section className="section-block" id="components">
          <div className="section-heading">
            <div>
              <div className="eyebrow">01 / RESOURCE MODULES</div>
              <h2>硬件资源总览</h2>
              <p>点击任意模块，查看详细指标、历史变化和原始数据。</p>
            </div>
            <div className="sample-count">
              <span className="status-dot online" />
              {history.length} 条历史快照
            </div>
          </div>

          <div className="metric-grid">
            {(
              ["gpu", "cpu", "memory", "disk", "network", "system"] as SectionKey[]
            ).map((section) => (
              <MetricCard
                key={section}
                section={section}
                snapshot={latest}
                onOpen={setSelectedSection}
              />
            ))}
          </div>
        </section>

        <section className="section-block" id="resource-trends">
          <div className="section-heading">
            <div>
              <div className="eyebrow">02 / TIME SERIES</div>
              <h2>资源变化趋势</h2>
              <p>根据数据库中的历史采样点，观察资源占用随时间的变化。</p>
            </div>
            <div className="live-label">
              <span className="live-pulse" />
              TELEMETRY
            </div>
          </div>

          <div className="trend-layout">
            <div className="panel trend-panel">
              <div className="panel-heading">
                <div>
                  <h3>系统资源使用率</h3>
                  <p>GPU · CPU · RAM · DISK</p>
                </div>
                <div className="panel-period">最近 {chartData.length} 个采样点</div>
              </div>

              {chartData.length === 0 ? (
                <EmptyState text="等待硬件采集器上传第一条快照。" />
              ) : (
                <div className="main-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={chartData}
                      margin={{ top: 10, right: 12, left: -18, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id="gpuFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#54E7FF" stopOpacity={0.32} />
                          <stop offset="100%" stopColor="#54E7FF" stopOpacity={0.01} />
                        </linearGradient>
                        <linearGradient id="cpuFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#A69AFF" stopOpacity={0.22} />
                          <stop offset="100%" stopColor="#A69AFF" stopOpacity={0.01} />
                        </linearGradient>
                        <linearGradient id="memoryFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#48E3B5" stopOpacity={0.2} />
                          <stop offset="100%" stopColor="#48E3B5" stopOpacity={0.01} />
                        </linearGradient>
                      </defs>

                      <CartesianGrid stroke="#1A3048" strokeDasharray="4 4" vertical={false} />

                      <XAxis
                        dataKey="time"
                        tick={{ fill: "#8199B5", fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                        minTickGap={28}
                      />

                      <YAxis
                        domain={[0, 100]}
                        tick={{ fill: "#8199B5", fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value: number) => `${value}%`}
                      />

                      <Tooltip
                        contentStyle={{
                          background: "#091525",
                          border: "1px solid #29425D",
                          borderRadius: 10,
                          color: "#EDF7FF",
                        }}
                      />

                      <Area
                        type="monotone"
                        dataKey="gpu"
                        name="GPU"
                        stroke="#54E7FF"
                        fill="url(#gpuFill)"
                        strokeWidth={2}
                        connectNulls
                      />

                      <Area
                        type="monotone"
                        dataKey="cpu"
                        name="CPU"
                        stroke="#A69AFF"
                        fill="url(#cpuFill)"
                        strokeWidth={2}
                        connectNulls
                      />

                      <Area
                        type="monotone"
                        dataKey="memory"
                        name="RAM"
                        stroke="#48E3B5"
                        fill="url(#memoryFill)"
                        strokeWidth={2}
                        connectNulls
                      />

                      <Line
                        type="monotone"
                        dataKey="disk"
                        name="磁盘"
                        stroke="#FFBD6B"
                        strokeWidth={1.8}
                        dot={false}
                        connectNulls
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}

              <div className="chart-legend">
                <span><i style={{ background: "#54E7FF" }} /> GPU 利用率</span>
                <span><i style={{ background: "#A69AFF" }} /> CPU 利用率</span>
                <span><i style={{ background: "#48E3B5" }} /> 内存利用率</span>
                <span><i style={{ background: "#FFBD6B" }} /> 磁盘使用率</span>
              </div>

              {chartData.length < 2 && (
                <div className="chart-note">
                  当前历史采样点不足，持续运行采集器后即可观察变化曲线。
                </div>
              )}
            </div>

            <aside className="panel health-panel">
              <div className="panel-heading">
                <div>
                  <h3>资源状态</h3>
                  <p>当前快照指标</p>
                </div>
                <ShieldCheck size={19} />
              </div>

              {[
                { name: "GPU", value: gpu.utilization_percent, color: "#54E7FF" },
                { name: "CPU", value: cpu.utilization_percent, color: "#A69AFF" },
                { name: "RAM", value: memory.percent, color: "#48E3B5" },
                { name: "DISK", value: disk?.percent, color: "#FFBD6B" },
              ].map((item) => (
                <div className="health-row" key={item.name}>
                  <div className="health-row-label">
                    <span>{item.name}</span>
                    <strong>{formatPercent(item.value)}</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${clampPercent(item.value)}%`,
                        background: item.color,
                      }}
                    />
                  </div>
                </div>
              ))}

              <div className={`health-summary ${health}`}>
                {health === "critical" ? (
                  <AlertTriangle size={18} />
                ) : health === "nominal" ? (
                  <CheckCircle2 size={18} />
                ) : (
                  <Activity size={18} />
                )}
                <div>
                  <strong>{healthText}</strong>
                  <p>基于当前快照的基础阈值判断</p>
                </div>
              </div>

              <div className="health-disclaimer">
                预警阈值是界面初始规则，不是硬件厂商的安全认证。
                实际判断需结合设备规格和训练负载。
              </div>
            </aside>
          </div>
        </section>

        <section className="section-block">
          <div className="section-heading">
            <div>
              <div className="eyebrow">03 / TELEMETRY PIPELINE</div>
              <h2>采集与数据链路</h2>
              <p>追踪硬件数据从采集器到数据库的完整路径。</p>
            </div>
          </div>

          <div className="pipeline-grid">
            <div className="pipeline-step">
              <div className="pipeline-icon"><Cpu size={20} /></div>
              <span className="pipeline-number">STEP 01</span>
              <strong>硬件采集器</strong>
              <p>psutil / nvidia-smi</p>
              <span className="pipeline-detail">采样时间：{formatDateTime(latest?.sampled_at)}</span>
            </div>

            <div className="pipeline-arrow"><ChevronRight size={20} /></div>

            <div className="pipeline-step">
              <div className="pipeline-icon"><Network size={20} /></div>
              <span className="pipeline-number">STEP 02</span>
              <strong>C++ Drogon API</strong>
              <p>/api/hardware-snapshots</p>
              <span className="pipeline-detail">{apiOnline ? "API 可访问" : "API 未连接"}</span>
            </div>

            <div className="pipeline-arrow"><ChevronRight size={20} /></div>

            <div className="pipeline-step">
              <div className="pipeline-icon"><Database size={20} /></div>
              <span className="pipeline-number">STEP 03</span>
              <strong>PostgreSQL</strong>
              <p>hardware_snapshots</p>
              <span className="pipeline-detail">当前读取 {history.length} 条历史记录</span>
            </div>

            <div className="pipeline-arrow"><ChevronRight size={20} /></div>

            <div className="pipeline-step">
              <div className="pipeline-icon"><Monitor size={20} /></div>
              <span className="pipeline-number">STEP 04</span>
              <strong>监控前端</strong>
              <p>React / Recharts</p>
              <span className="pipeline-detail">
                最近刷新：{lastUpdated ? formatClock(lastUpdated.toISOString()) : "等待连接"}
              </span>
            </div>
          </div>
        </section>

        <TrainingPanel />

        <footer className="app-footer">
          <div>
            <span className="footer-mark">SLM</span>
            <span>FROM SCRATCH · HARDWARE OBSERVABILITY</span>
          </div>
          <div>刷新间隔 5 秒 · 所有图表基于已采集数据</div>
        </footer>
      </main>

      {selectedSection && (
        <DetailModal
          section={selectedSection}
          snapshot={latest}
          chartData={chartData}
          onClose={() => setSelectedSection(null)}
        />
      )}
    </div>
  );
}

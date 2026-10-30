import { numeric } from "./training-data";

export const number = (n: unknown, digits = 3) =>
  numeric(n) === null
    ? "—"
    : Number(n).toLocaleString("zh-CN", { maximumFractionDigits: digits });
export const scientific = (n: unknown) =>
  numeric(n) === null ? "—" : Number(n).toExponential(2);
export const token = (value: string) =>
  value === " " ? "␠" : value === "\n" ? "↵" : value === "\t" ? "⇥" : value;
export const dateText = (value?: string) =>
  value && Number.isFinite(Date.parse(value))
    ? new Date(value).toLocaleString("zh-CN", { hour12: false })
    : "—";

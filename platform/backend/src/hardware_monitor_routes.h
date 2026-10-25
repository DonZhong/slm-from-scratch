
#pragma once

#include <drogon/drogon.h>
#include <drogon/orm/DbClient.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <regex>
#include <string>

namespace slm_hardware {

using Reply =
    std::function<void(const drogon::HttpResponsePtr&)>;

// 统一返回 JSON 错误
template <typename Status>
void sendError(
    const Reply& reply,
    Status status,
    const std::string& message
) {
    Json::Value body;
    body["status"] = "error";
    body["message"] = message;

    auto response =
        drogon::HttpResponse::newHttpJsonResponse(body);

    response->setStatusCode(status);
    reply(response);
}

// 校验正整数 ID
inline bool parsePositiveInt64(
    const std::string& text,
    int64_t& value
) {
    try {
        std::size_t parsed = 0;
        const auto number = std::stoll(text, &parsed);

        if (parsed != text.size() || number <= 0) {
            return false;
        }

        value = static_cast<int64_t>(number);
        return true;
    } catch (...) {
        return false;
    }
}

// 将数据库中的 JSONB 字段转换回 JSON 对象
inline Json::Value parseMetrics(
    const std::string& text
) {
    Json::Reader reader;
    Json::Value metrics;

    if (!reader.parse(text, metrics) || !metrics.isObject()) {
        return Json::Value(Json::objectValue);
    }

    return metrics;
}

// 将数据库记录转换成前端使用的 JSON 结构
template <typename Row>
Json::Value rowToJson(const Row& row) {
    Json::Value item;

    item["id"] = static_cast<Json::Int64>(
        row["id"].template as<int64_t>()
    );

    item["training_run_id"] =
        row["training_run_id"].isNull()
        ? Json::Value(Json::nullValue)
        : Json::Value(static_cast<Json::Int64>(
            row["training_run_id"].template as<int64_t>()
        ));

    item["sampled_at"] =
        row["sampled_at"].template as<std::string>();

    item["host_name"] =
        row["host_name"].template as<std::string>();

    item["received_at"] =
        row["received_at"].template as<std::string>();

    item["metrics"] = parseMetrics(
        row["metrics_json"].template as<std::string>()
    );

    return item;
}

template <typename DbClientType>
void registerHardwareMonitorRoutes(
    const DbClientType& dbClient
) {
    // =========================================================
    // POST /api/hardware-snapshots
    // 接收并保存一份硬件快照
    // =========================================================
    drogon::app().registerHandler(
        "/api/hardware-snapshots",
        [dbClient](
            const drogon::HttpRequestPtr& request,
            Reply&& callback
        ) {
            auto reply = std::move(callback);

            const auto json = request->getJsonObject();

            if (!json || !json->isObject()) {
                sendError(
                    reply,
                    drogon::k400BadRequest,
                    "A JSON object is required"
                );
                return;
            }

            const auto& data = *json;

            if (
                !data.isMember("sampled_at") ||
                !data["sampled_at"].isString() ||
                !data.isMember("host_name") ||
                !data["host_name"].isString() ||
                !data.isMember("metrics") ||
                !data["metrics"].isObject()
            ) {
                sendError(
                    reply,
                    drogon::k400BadRequest,
                    "sampled_at, host_name and metrics are required"
                );
                return;
            }

            const std::string sampledAt =
                data["sampled_at"].asString();

            const std::string hostName =
                data["host_name"].asString();

            const Json::Value metrics = data["metrics"];

            // 要求带时区的 ISO 8601 时间，例如 2026-10-25T13:00:00Z
            static const std::regex timestampPattern(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$)"
            );

            if (!std::regex_match(sampledAt, timestampPattern)) {
                sendError(
                    reply,
                    drogon::k400BadRequest,
                    "sampled_at must be a timezone-aware ISO 8601 timestamp"
                );
                return;
            }

            if (
                hostName.find_first_not_of(" \t\r\n") ==
                std::string::npos
            ) {
                sendError(
                    reply,
                    drogon::k400BadRequest,
                    "host_name must not be empty"
                );
                return;
            }

            // training_run_id 可省略或传 null
            std::optional<int64_t> trainingRunId;

            if (
                data.isMember("training_run_id") &&
                !data["training_run_id"].isNull()
            ) {
                if (!data["training_run_id"].isInt64()) {
                    sendError(
                        reply,
                        drogon::k400BadRequest,
                        "training_run_id must be a positive integer or null"
                    );
                    return;
                }

                const int64_t id =
                    data["training_run_id"].asInt64();

                if (id <= 0) {
                    sendError(
                        reply,
                        drogon::k400BadRequest,
                        "training_run_id must be positive"
                    );
                    return;
                }

                trainingRunId = id;
            }

            const std::string metricsJson =
                metrics.toStyledString();

            // 实际写入数据库的操作
            auto insertSnapshot = [
                dbClient,
                reply,
                sampledAt,
                hostName,
                metrics,
                metricsJson,
                trainingRunId
            ]() {
                auto onSuccess = [
                    reply,
                    sampledAt,
                    hostName,
                    metrics,
                    trainingRunId
                ](const drogon::orm::Result& result) {
                    Json::Value item;

                    item["id"] = static_cast<Json::Int64>(
                        result[0]["id"].as<int64_t>()
                    );

                    item["training_run_id"] =
                        trainingRunId
                        ? Json::Value(static_cast<Json::Int64>(
                            *trainingRunId
                        ))
                        : Json::Value(Json::nullValue);

                    item["sampled_at"] =
                        result[0]["sampled_at"].as<std::string>();

                    item["host_name"] = hostName;
                    item["metrics"] = metrics;

                    item["received_at"] =
                        result[0]["received_at"].as<std::string>();

                    Json::Value body;
                    body["status"] = "ok";
                    body["item"] = item;

                    auto response =
                        drogon::HttpResponse::newHttpJsonResponse(body);

                    response->setStatusCode(drogon::k201Created);
                    reply(response);
                };

                auto onError = [reply](
                    const drogon::orm::DrogonDbException&
                ) {
                    LOG_ERROR << "Failed to insert hardware snapshot";

                    sendError(
                        reply,
                        drogon::k503ServiceUnavailable,
                        "Unable to save hardware snapshot"
                    );
                };

                if (trainingRunId) {
                    dbClient->execSqlAsync(
                        "INSERT INTO hardware_snapshots "
                        "(training_run_id, sampled_at, host_name, metrics) "
                        "VALUES ($1, $2::timestamptz, $3, $4::jsonb) "
                        "RETURNING id, sampled_at::text AS sampled_at, "
                        "received_at::text AS received_at",
                        onSuccess,
                        onError,
                        *trainingRunId,
                        sampledAt,
                        hostName,
                        metricsJson
                    );
                } else {
                    dbClient->execSqlAsync(
                        "INSERT INTO hardware_snapshots "
                        "(training_run_id, sampled_at, host_name, metrics) "
                        "VALUES (NULL, $1::timestamptz, $2, $3::jsonb) "
                        "RETURNING id, sampled_at::text AS sampled_at, "
                        "received_at::text AS received_at",
                        onSuccess,
                        onError,
                        sampledAt,
                        hostName,
                        metricsJson
                    );
                }
            };

            // 如果关联了训练实验，先确认实验确实存在
            if (trainingRunId) {
                dbClient->execSqlAsync(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM training_runs WHERE id = $1"
                    ") AS run_exists",

                    [reply, insertSnapshot](
                        const drogon::orm::Result& result
                    ) {
                        if (!result[0]["run_exists"].as<bool>()) {
                            sendError(
                                reply,
                                drogon::k404NotFound,
                                "Training run not found"
                            );
                            return;
                        }

                        insertSnapshot();
                    },

                    [reply](
                        const drogon::orm::DrogonDbException&
                    ) {
                        sendError(
                            reply,
                            drogon::k503ServiceUnavailable,
                            "Unable to validate training run"
                        );
                    },

                    *trainingRunId
                );
            } else {
                insertSnapshot();
            }
        },
        {drogon::Post}
    );

    // =========================================================
    // GET /api/hardware-snapshots/latest
    // 查询最新硬件快照
    // 可选：?training_run_id=12
    // =========================================================
    drogon::app().registerHandler(
        "/api/hardware-snapshots/latest",
        [dbClient](
            const drogon::HttpRequestPtr& request,
            Reply&& callback
        ) {
            auto reply = std::move(callback);

            std::optional<int64_t> runId;
            const std::string runIdText =
                request->getParameter("training_run_id");

            if (!runIdText.empty()) {
                int64_t parsedId = 0;

                if (!parsePositiveInt64(runIdText, parsedId)) {
                    sendError(
                        reply,
                        drogon::k400BadRequest,
                        "Invalid training_run_id"
                    );
                    return;
                }

                runId = parsedId;
            }

            auto onSuccess = [reply](
                const drogon::orm::Result& result
            ) {
                Json::Value body;
                body["status"] = "ok";
                body["count"] =
                    static_cast<Json::Int64>(result.size());

                if (result.empty()) {
                    body["item"] = Json::Value(Json::nullValue);
                } else {
                    body["item"] = rowToJson(result[0]);
                }

                reply(
                    drogon::HttpResponse::newHttpJsonResponse(body)
                );
            };

            auto onError = [reply](
                const drogon::orm::DrogonDbException&
            ) {
                LOG_ERROR << "Failed to query latest hardware snapshot";

                sendError(
                    reply,
                    drogon::k503ServiceUnavailable,
                    "Unable to retrieve hardware snapshot"
                );
            };

            if (runId) {
                dbClient->execSqlAsync(
                    "SELECT id, training_run_id, "
                    "sampled_at::text AS sampled_at, host_name, "
                    "metrics::text AS metrics_json, "
                    "received_at::text AS received_at "
                    "FROM hardware_snapshots "
                    "WHERE training_run_id = $1 "
                    "ORDER BY sampled_at DESC, id DESC LIMIT 1",
                    onSuccess,
                    onError,
                    *runId
                );
            } else {
                dbClient->execSqlAsync(
                    "SELECT id, training_run_id, "
                    "sampled_at::text AS sampled_at, host_name, "
                    "metrics::text AS metrics_json, "
                    "received_at::text AS received_at "
                    "FROM hardware_snapshots "
                    "ORDER BY sampled_at DESC, id DESC LIMIT 1",
                    onSuccess,
                    onError
                );
            }
        },
        {drogon::Get}
    );

    // =========================================================
    // GET /api/hardware-snapshots
    // 查询历史快照，按时间正序返回，方便前端绘图
    // 参数：limit（默认 100，最大 500）
    // 可选：training_run_id
    // =========================================================
    drogon::app().registerHandler(
        "/api/hardware-snapshots",
        [dbClient](
            const drogon::HttpRequestPtr& request,
            Reply&& callback
        ) {
            auto reply = std::move(callback);

            int64_t limit = 100;
            const std::string limitText =
                request->getParameter("limit");

            if (!limitText.empty()) {
                if (!parsePositiveInt64(limitText, limit)) {
                    sendError(
                        reply,
                        drogon::k400BadRequest,
                        "limit must be a positive integer"
                    );
                    return;
                }

                if (limit > 500) {
                    sendError(
                        reply,
                        drogon::k400BadRequest,
                        "limit must not exceed 500"
                    );
                    return;
                }
            }

            std::optional<int64_t> runId;
            const std::string runIdText =
                request->getParameter("training_run_id");

            if (!runIdText.empty()) {
                int64_t parsedId = 0;

                if (!parsePositiveInt64(runIdText, parsedId)) {
                    sendError(
                        reply,
                        drogon::k400BadRequest,
                        "Invalid training_run_id"
                    );
                    return;
                }

                runId = parsedId;
            }

            auto onSuccess = [reply, limit](
                const drogon::orm::Result& result
            ) {
                Json::Value body;
                body["status"] = "ok";
                body["count"] =
                    static_cast<Json::Int64>(result.size());
                body["limit"] = static_cast<Json::Int64>(limit);
                body["items"] = Json::Value(Json::arrayValue);

                for (const auto& row : result) {
                    body["items"].append(rowToJson(row));
                }

                reply(
                    drogon::HttpResponse::newHttpJsonResponse(body)
                );
            };

            auto onError = [reply](
                const drogon::orm::DrogonDbException&
            ) {
                LOG_ERROR << "Failed to query hardware snapshots";

                sendError(
                    reply,
                    drogon::k503ServiceUnavailable,
                    "Unable to retrieve hardware snapshots"
                );
            };

            if (runId) {
                dbClient->execSqlAsync(
                    "SELECT id, training_run_id, "
                    "sampled_at::text AS sampled_at, host_name, "
                    "metrics_json, received_at::text AS received_at "
                    "FROM ("
                    "SELECT id, training_run_id, sampled_at, host_name, "
                    "metrics::text AS metrics_json, received_at "
                    "FROM hardware_snapshots "
                    "WHERE training_run_id = $1 "
                    "ORDER BY sampled_at DESC, id DESC LIMIT $2"
                    ") AS recent "
                    "ORDER BY sampled_at ASC, id ASC",
                    onSuccess,
                    onError,
                    *runId,
                    limit
                );
            } else {
                dbClient->execSqlAsync(
                    "SELECT id, training_run_id, "
                    "sampled_at::text AS sampled_at, host_name, "
                    "metrics_json, received_at::text AS received_at "
                    "FROM ("
                    "SELECT id, training_run_id, sampled_at, host_name, "
                    "metrics::text AS metrics_json, received_at "
                    "FROM hardware_snapshots "
                    "ORDER BY sampled_at DESC, id DESC LIMIT $1"
                    ") AS recent "
                    "ORDER BY sampled_at ASC, id ASC",
                    onSuccess,
                    onError,
                    limit
                );
            }
        },
        {drogon::Get}
    );
}

}  // namespace slm_hardware

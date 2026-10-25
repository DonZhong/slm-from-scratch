#include <cmath>
#include <optional>
#include "hardware_monitor_routes.h"
#include <drogon/drogon.h>
#include <drogon/orm/DbClient.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

// 读取必需的环境变量
std::string requireEnv(const char* name) {
    const char* value = std::getenv(name);

    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(
            std::string("Missing environment variable: ") + name
        );
    }

    return value;
}

// 安全地引用 PostgreSQL 连接字符串中的字段值
std::string quotePgValue(const std::string& value) {
    std::string quoted = "'";

    for (char ch : value) {
        if (ch == '\\' || ch == '\'') {
            quoted.push_back('\\');
        }

        quoted.push_back(ch);
    }

    quoted.push_back('\'');
    return quoted;
}

bool parsePositiveInt64(
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

int main() {
    // 读取数据库配置
    const std::string dbName = requireEnv("SLM_DB_NAME");
    const std::string dbUser = requireEnv("SLM_DB_USER");
    const std::string dbPassword = requireEnv("SLM_DB_PASSWORD");

    // 在 Docker 网络中通过服务名 postgres 访问数据库
    const std::string connInfo =
        "host=postgres port=5432 dbname=" + quotePgValue(dbName) +
        " user=" + quotePgValue(dbUser) +
        " password=" + quotePgValue(dbPassword) +
        " connect_timeout=5";

    // 创建数据库客户端，并保持其生命周期
    auto dbClient = drogon::orm::DbClient::newPgClient(connInfo, 1);

    // 原有后端健康检查
    drogon::app().registerHandler(
        "/api/health",
        [](const drogon::HttpRequestPtr&,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            Json::Value body;
            body["status"] = "ok";
            body["service"] = "slm-platform-backend";

            callback(drogon::HttpResponse::newHttpJsonResponse(body));
        },
        {drogon::Get}
    );

    // 新增数据库健康检查
    drogon::app().registerHandler(
        "/api/db/health",
        [dbClient](
            const drogon::HttpRequestPtr&,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback
        ) {
            auto reply = std::move(callback);

            dbClient->execSqlAsync(
                "SELECT COUNT(*) AS run_count FROM training_runs",

                [reply](const drogon::orm::Result& result) {
                    Json::Value body;
                    body["status"] = "ok";
                    body["database"] = "connected";
                    body["training_run_count"] =
                        static_cast<Json::Int64>(
                            result[0]["run_count"].as<int64_t>()
                        );

                    reply(drogon::HttpResponse::newHttpJsonResponse(body));
                },

                [reply](const drogon::orm::DrogonDbException&) {
                    LOG_ERROR << "Database health query failed";

                    Json::Value body;
                    body["status"] = "error";
                    body["database"] = "unavailable";

                    auto response =
                        drogon::HttpResponse::newHttpJsonResponse(body);

                    response->setStatusCode(
                        drogon::k503ServiceUnavailable
                    );

                    reply(response);
                }
            );
        },
        {drogon::Get}
    );


    // 训练实验列表 API
    drogon::app().registerHandler(
        "/api/training-runs",
        [dbClient](
            const drogon::HttpRequestPtr&,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback
        ) {
            auto reply = std::move(callback);

            dbClient->execSqlAsync(
                "SELECT id, name, model_name, dataset_name, status, "
                "created_at::text AS created_at, "
                "COALESCE(started_at::text, '') AS started_at, "
                "COALESCE(finished_at::text, '') AS finished_at "
                "FROM training_runs "
                "ORDER BY id DESC LIMIT 100",

                [reply](
                    const drogon::orm::Result& result
                ) {
                    Json::Value body;
                    body["status"] = "ok";
                    body["count"] =
                        static_cast<Json::Int64>(result.size());
                    body["items"] = Json::Value(Json::arrayValue);

                    for (const auto& row : result) {
                        Json::Value item;

                        item["id"] = static_cast<Json::Int64>(
                            row["id"].as<int64_t>()
                        );
                        item["name"] =
                            row["name"].as<std::string>();
                        item["model_name"] =
                            row["model_name"].as<std::string>();
                        item["dataset_name"] =
                            row["dataset_name"].as<std::string>();
                        item["status"] =
                            row["status"].as<std::string>();
                        item["created_at"] =
                            row["created_at"].as<std::string>();
                        item["started_at"] =
                            row["started_at"].as<std::string>();
                        item["finished_at"] =
                            row["finished_at"].as<std::string>();

                        body["items"].append(item);
                    }

                    reply(
                        drogon::HttpResponse::newHttpJsonResponse(body)
                    );
                },

                [reply](
                    const drogon::orm::DrogonDbException&
                ) {
                    LOG_ERROR << "Failed to query training runs";

                    Json::Value body;
                    body["status"] = "error";
                    body["message"] =
                        "Unable to retrieve training runs";

                    auto response =
                        drogon::HttpResponse::newHttpJsonResponse(body);

                    response->setStatusCode(
                        drogon::k503ServiceUnavailable
                    );

                    reply(response);
                }
            );
        },
        {drogon::Get}
    );

// POST /api/training-runs/{run_id}/metrics
// 写入一次训练指标
drogon::app().registerHandler(
    "/api/training-runs/{run_id}/metrics",
    [dbClient](
        const drogon::HttpRequestPtr& request,
        std::function<void(
            const drogon::HttpResponsePtr&
        )>&& callback,
        const std::string& runIdText
    ) {
        // 统一返回 JSON 错误
        auto sendError = [&callback](
            auto status,
            const std::string& message
        ) {
            Json::Value body;
            body["status"] = "error";
            body["message"] = message;

            auto response =
                drogon::HttpResponse::newHttpJsonResponse(body);

            response->setStatusCode(status);
            callback(response);
        };

        // 1. 校验实验 ID
        int64_t runId = 0;

        if (!parsePositiveInt64(runIdText, runId)) {
            sendError(
                drogon::k400BadRequest,
                "Invalid training run ID"
            );
            return;
        }

        // 2. 读取 JSON 请求体
        const auto json = request->getJsonObject();

        if (!json || !json->isObject()) {
            sendError(
                drogon::k400BadRequest,
                "A JSON object is required"
            );
            return;
        }

        const auto& data = *json;

        // 必需字段必须存在且类型正确
        if (
            !data.isMember("global_step") ||
            !data["global_step"].isInt64() ||
            !data.isMember("epoch") ||
            !data["epoch"].isInt() ||
            !data.isMember("train_loss") ||
            !data["train_loss"].isNumeric() ||
            !data.isMember("learning_rate") ||
            !data["learning_rate"].isNumeric()
        ) {
            sendError(
                drogon::k400BadRequest,
                "Missing or invalid required fields"
            );
            return;
        }

        const int64_t globalStep =
            data["global_step"].asInt64();

        const int epoch = data["epoch"].asInt();

        const double trainLoss =
            data["train_loss"].asDouble();

        const double learningRate =
            data["learning_rate"].asDouble();

        if (
            globalStep <= 0 ||
            epoch <= 0 ||
            !std::isfinite(trainLoss) ||
            trainLoss < 0 ||
            !std::isfinite(learningRate) ||
            learningRate <= 0
        ) {
            sendError(
                drogon::k400BadRequest,
                "Metric values are outside valid ranges"
            );
            return;
        }

        // 3. 可选字段：缺失或 JSON null 均存储为 SQL NULL
        std::optional<double> valLoss;
        std::optional<double> tokensPerSecond;

        if (
            data.isMember("val_loss") &&
            !data["val_loss"].isNull()
        ) {
            if (!data["val_loss"].isNumeric()) {
                sendError(
                    drogon::k400BadRequest,
                    "val_loss must be numeric or null"
                );
                return;
            }

            valLoss = data["val_loss"].asDouble();

            if (
                !std::isfinite(*valLoss) ||
                *valLoss < 0
            ) {
                sendError(
                    drogon::k400BadRequest,
                    "Invalid val_loss"
                );
                return;
            }
        }

        if (
            data.isMember("tokens_per_second") &&
            !data["tokens_per_second"].isNull()
        ) {
            if (!data["tokens_per_second"].isNumeric()) {
                sendError(
                    drogon::k400BadRequest,
                    "tokens_per_second must be numeric or null"
                );
                return;
            }

            tokensPerSecond =
                data["tokens_per_second"].asDouble();

            if (
                !std::isfinite(*tokensPerSecond) ||
                *tokensPerSecond < 0
            ) {
                sendError(
                    drogon::k400BadRequest,
                    "Invalid tokens_per_second"
                );
                return;
            }
        }

        // 4. 准备异步数据库操作
        auto reply = std::move(callback);

        auto onSuccess = [
            reply,
            runId,
            globalStep,
            epoch,
            trainLoss,
            valLoss,
            learningRate,
            tokensPerSecond
        ](const drogon::orm::Result& result) {
                        // ON CONFLICT DO NOTHING 时，
            // 如果没有插入新记录，RETURNING id 会返回空结果。
            if (result.size() == 0) {
                Json::Value body;
                body["status"] = "error";
                body["message"] =
                    "Metric already exists for this training step";

                auto response =
                    drogon::HttpResponse::newHttpJsonResponse(body);

                response->setStatusCode(drogon::k409Conflict);
                reply(response);
                return;
            }
            Json::Value item;

            item["id"] = static_cast<Json::Int64>(
                result[0]["id"].as<int64_t>()
            );
            item["run_id"] =
                static_cast<Json::Int64>(runId);
            item["global_step"] =
                static_cast<Json::Int64>(globalStep);
            item["epoch"] = epoch;
            item["train_loss"] = trainLoss;
            item["val_loss"] = valLoss
                ? Json::Value(*valLoss)
                : Json::Value(Json::nullValue);
            item["learning_rate"] = learningRate;
            item["tokens_per_second"] = tokensPerSecond
                ? Json::Value(*tokensPerSecond)
                : Json::Value(Json::nullValue);

            Json::Value body;
            body["status"] = "ok";
            body["item"] = item;

            auto response =
                drogon::HttpResponse::newHttpJsonResponse(body);

            response->setStatusCode(drogon::k201Created);
            reply(response);
        };


        auto onError = [dbClient, reply, runId](
            const drogon::orm::DrogonDbException&
        ) {
            LOG_ERROR << "Failed to insert training metric";

            // 插入失败后，检查对应的训练实验是否存在。
            dbClient->execSqlAsync(
                "SELECT EXISTS ("
                "SELECT 1 FROM training_runs WHERE id = $1"
                ") AS run_exists",

                [reply](
                    const drogon::orm::Result& result
                ) {
                    const bool runExists =
                        result[0]["run_exists"].as<bool>();

                    Json::Value body;
                    body["status"] = "error";

                    if (runExists) {
                        body["message"] =
                            "Unable to save training metric";
                    } else {
                        body["message"] =
                            "Training run not found";
                    }

                    auto response =
                        drogon::HttpResponse::newHttpJsonResponse(body);

                    if (runExists) {
                        response->setStatusCode(
                            drogon::k500InternalServerError
                        );
                    } else {
                        response->setStatusCode(
                            drogon::k404NotFound
                        );
                    }

                    reply(response);
                },

                [reply](
                    const drogon::orm::DrogonDbException&
                ) {
                    LOG_ERROR
                        << "Failed to check training run after metric insert error";

                    Json::Value body;
                    body["status"] = "error";
                    body["message"] =
                        "Unable to save training metric";

                    auto response =
                        drogon::HttpResponse::newHttpJsonResponse(body);

                    response->setStatusCode(
                        drogon::k500InternalServerError
                    );

                    reply(response);
                },

                runId
            );
        };


        // 5. 按可选字段情况选择固定 SQL。
        // SQL 使用绑定参数，不拼接用户提供的数值。
        if (valLoss && tokensPerSecond) {
            dbClient->execSqlAsync(
                "INSERT INTO training_metrics "
                "(run_id, global_step, epoch, train_loss, "
                "val_loss, learning_rate, tokens_per_second) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7) "
                "ON CONFLICT (run_id, global_step) DO NOTHING "
                "RETURNING id",
                onSuccess,
                onError,
                runId,
                globalStep,
                epoch,
                trainLoss,
                *valLoss,
                learningRate,
                *tokensPerSecond
            );
        } else if (valLoss && !tokensPerSecond) {
            dbClient->execSqlAsync(
                "INSERT INTO training_metrics "
                "(run_id, global_step, epoch, train_loss, "
                "val_loss, learning_rate, tokens_per_second) "
                "VALUES ($1, $2, $3, $4, $5, $6, NULL) "
                "ON CONFLICT (run_id, global_step) DO NOTHING "
                "RETURNING id",
                onSuccess,
                onError,
                runId,
                globalStep,
                epoch,
                trainLoss,
                *valLoss,
                learningRate
            );
        } else if (!valLoss && tokensPerSecond) {
            dbClient->execSqlAsync(
                "INSERT INTO training_metrics "
                "(run_id, global_step, epoch, train_loss, "
                "val_loss, learning_rate, tokens_per_second) "
                "VALUES ($1, $2, $3, $4, NULL, $5, $6) "
                "ON CONFLICT (run_id, global_step) DO NOTHING "
                "RETURNING id",
                onSuccess,
                onError,
                runId,
                globalStep,
                epoch,
                trainLoss,
                learningRate,
                *tokensPerSecond
            );
        } else {
            dbClient->execSqlAsync(
                "INSERT INTO training_metrics "
                "(run_id, global_step, epoch, train_loss, "
                "val_loss, learning_rate, tokens_per_second) "
                "VALUES ($1, $2, $3, $4, NULL, $5, NULL) "
                "ON CONFLICT (run_id, global_step) DO NOTHING "
                "RETURNING id",
                onSuccess,
                onError,
                runId,
                globalStep,
                epoch,
                trainLoss,
                learningRate
            );
        }
    },
    {drogon::Post}
);


    // GET /api/training-runs/{run_id}/metrics
    // 查询一次训练实验的历史指标
    drogon::app().registerHandler(
        "/api/training-runs/{run_id}/metrics",
        [dbClient](
            const drogon::HttpRequestPtr&,
            std::function<void(
                const drogon::HttpResponsePtr&
            )>&& callback,
            const std::string& runIdText
        ) {
            auto reply = std::move(callback);

            // 统一返回 JSON 错误
            auto sendError = [reply](
                auto status,
                const std::string& message
            ) {
                Json::Value body;
                body["status"] = "error";
                body["message"] = message;

                auto response =
                    drogon::HttpResponse::newHttpJsonResponse(body);

                response->setStatusCode(status);
                reply(response);
            };

            // 1. 校验训练实验 ID
            int64_t runId = 0;

            if (!parsePositiveInt64(runIdText, runId)) {
                sendError(
                    drogon::k400BadRequest,
                    "Invalid training run ID"
                );
                return;
            }

            // 2. 先确认训练实验存在
            dbClient->execSqlAsync(
                "SELECT id FROM training_runs WHERE id = $1",

                [dbClient, reply, sendError, runId](
                    const drogon::orm::Result& runResult
                ) {
                    if (runResult.empty()) {
                        sendError(
                            drogon::k404NotFound,
                            "Training run not found"
                        );
                        return;
                    }

                    // 3. 查询最近 1000 条指标，
                    //    最终按 global_step 从小到大返回。
                    dbClient->execSqlAsync(
                        "SELECT id, global_step, epoch, train_loss, "
                        "val_loss, learning_rate, tokens_per_second "
                        "FROM ("
                        " SELECT id, global_step, epoch, train_loss, "
                        " val_loss, learning_rate, tokens_per_second "
                        " FROM training_metrics "
                        " WHERE run_id = $1 "
                        " ORDER BY global_step DESC "
                        " LIMIT 1000"
                        ") AS recent_metrics "
                        "ORDER BY global_step ASC",

                        [reply, runId](
                            const drogon::orm::Result& result
                        ) {
                            Json::Value body;
                            body["status"] = "ok";
                            body["run_id"] =
                                static_cast<Json::Int64>(runId);
                            body["count"] =
                                static_cast<Json::Int64>(result.size());
                            body["items"] =
                                Json::Value(Json::arrayValue);

                            // 4. 将数据库记录转换成 JSON
                            for (const auto& row : result) {
                                Json::Value item;

                                item["id"] =
                                    static_cast<Json::Int64>(
                                        row["id"].as<int64_t>()
                                    );

                                item["global_step"] =
                                    static_cast<Json::Int64>(
                                        row["global_step"].as<int64_t>()
                                    );

                                item["epoch"] =
                                    row["epoch"].as<int>();

                                item["train_loss"] =
                                    row["train_loss"].as<double>();

                                item["val_loss"] =
                                    row["val_loss"].isNull()
                                    ? Json::Value(Json::nullValue)
                                    : Json::Value(
                                        row["val_loss"].as<double>()
                                    );

                                item["learning_rate"] =
                                    row["learning_rate"].as<double>();

                                item["tokens_per_second"] =
                                    row["tokens_per_second"].isNull()
                                    ? Json::Value(Json::nullValue)
                                    : Json::Value(
                                        row["tokens_per_second"].as<double>()
                                    );

                                body["items"].append(item);
                            }

                            reply(
                                drogon::HttpResponse::newHttpJsonResponse(
                                    body
                                )
                            );
                        },

                        [sendError](
                            const drogon::orm::DrogonDbException&
                        ) {
                            LOG_ERROR
                                << "Failed to query training metrics";

                            sendError(
                                drogon::k503ServiceUnavailable,
                                "Unable to retrieve training metrics"
                            );
                        },

                        runId
                    );
                },

                [sendError](
                    const drogon::orm::DrogonDbException&
                ) {
                    LOG_ERROR
                        << "Failed to check training run";

                    sendError(
                        drogon::k503ServiceUnavailable,
                        "Unable to retrieve training metrics"
                    );
                },

                runId
            );
        },
        {drogon::Get}
    );

    slm_hardware::registerHardwareMonitorRoutes(dbClient);
    drogon::app().addListener("0.0.0.0", 8080).run();

    return 0;
}

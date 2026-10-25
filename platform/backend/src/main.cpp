
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


    drogon::app().addListener("0.0.0.0", 8080).run();

    return 0;
}

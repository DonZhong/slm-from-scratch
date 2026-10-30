#pragma once
#include "hardware_monitor_routes.h"
#include <limits>

namespace slm_monitor {
using Reply = slm_hardware::Reply;
using Client = drogon::orm::DbClientPtr;
inline Json::Value nullable(const drogon::orm::Row& row, const char* key) {
    return row[key].isNull() ? Json::Value(Json::nullValue) : Json::Value(row[key].as<double>());
}
inline void registerTrainingMonitorRoutes(const Client& db) {
    drogon::app().registerHandler("/api/training-runs/{run_id}/runtime",
        [db](const drogon::HttpRequestPtr& request, Reply&& callback, const std::string& id) {
            auto reply = std::move(callback);
            int64_t runId;
            const auto data = request->getJsonObject();
            if (!slm_hardware::parsePositiveInt64(id, runId) || !data || !data->isObject() ||
                !(*data)["phase"].isString() || !(*data)["global_step"].isInt64() ||
                (*data)["global_step"].asInt64() < 0 || !(*data)["epoch"].isInt() ||
                (*data)["epoch"].asInt() < 0) {
                slm_hardware::sendError(reply, drogon::k400BadRequest, "phase, nonnegative global_step and epoch are required"); return;
            }
            const auto phase = (*data)["phase"].asString();
            if (phase != "training" && phase != "validation" && phase != "checkpoint" && phase != "completed" && phase != "failed") {
                slm_hardware::sendError(reply, drogon::k400BadRequest, "Unsupported runtime phase"); return;
            }
            const auto text = data->toStyledString();
            if (text.size() > 8192) {
                slm_hardware::sendError(reply, drogon::k400BadRequest, "runtime must not exceed 8 KiB"); return;
            }
            db->execSqlAsync("UPDATE training_runs SET runtime=$2::jsonb, monitor_updated_at=NOW() "
                "WHERE id=$1 AND status='running' RETURNING id",
                [reply](const drogon::orm::Result& result) {
                    if (result.empty()) { slm_hardware::sendError(reply, drogon::k409Conflict, "Run must exist and be running"); return; }
                    Json::Value body; body["status"]="ok"; reply(drogon::HttpResponse::newHttpJsonResponse(body));
                }, [reply](const drogon::orm::DrogonDbException&) {
                    slm_hardware::sendError(reply, drogon::k503ServiceUnavailable, "Unable to save runtime");
                }, runId, text);
        }, {drogon::Post});

    drogon::app().registerHandler("/api/training-runs/{run_id}/monitor",
        [db](const drogon::HttpRequestPtr&, Reply&& callback, const std::string& id) {
            auto reply=std::move(callback); int64_t runId;
            if (!slm_hardware::parsePositiveInt64(id, runId)) {
                slm_hardware::sendError(reply,drogon::k400BadRequest,"Invalid run ID"); return;
            }
            db->execSqlAsync(
                "SELECT r.status, r.config::text AS config, r.runtime::text AS runtime, "
                "COALESCE(r.monitor_updated_at::text,'') AS updated_at, s.*, f.train_loss AS first_loss, "
                "l.train_loss AS last_loss, v.val_loss AS last_val_loss, d.global_step AS diagnostic_step, "
                "d.diagnostics::text AS diagnostics FROM training_runs r "
                "LEFT JOIN LATERAL (SELECT COUNT(*) AS total_metrics, COALESCE(MAX(global_step),0) AS latest_step, "
                "MIN(train_loss) AS best_loss, AVG(tokens_per_second) AS average_speed, "
                "COUNT(diagnostics) AS diagnostics_count FROM training_metrics WHERE run_id=r.id) s ON true "
                "LEFT JOIN LATERAL (SELECT train_loss FROM training_metrics WHERE run_id=r.id ORDER BY global_step LIMIT 1) f ON true "
                "LEFT JOIN LATERAL (SELECT train_loss FROM training_metrics WHERE run_id=r.id ORDER BY global_step DESC LIMIT 1) l ON true "
                "LEFT JOIN LATERAL (SELECT val_loss FROM training_metrics WHERE run_id=r.id AND val_loss IS NOT NULL ORDER BY global_step DESC LIMIT 1) v ON true "
                "LEFT JOIN LATERAL (SELECT global_step, diagnostics FROM training_metrics WHERE run_id=r.id AND diagnostics IS NOT NULL ORDER BY global_step DESC LIMIT 1) d ON true "
                "WHERE r.id=$1",
                [reply,runId](const drogon::orm::Result& result) {
                    if (result.empty()) { slm_hardware::sendError(reply,drogon::k404NotFound,"Run not found"); return; }
                    const auto row=result[0]; Json::Value body; body["status"]="ok"; body["run_id"]=Json::Int64(runId);
                    body["run_status"]=row["status"].as<std::string>();
                    body["config"]=slm_hardware::parseMetrics(row["config"].as<std::string>());
                    body["runtime"]=slm_hardware::parseMetrics(row["runtime"].as<std::string>());
                    body["updated_at"]=row["updated_at"].as<std::string>();
                    auto& summary=body["summary"];
                    for (auto key : {"total_metrics","latest_step","diagnostics_count"}) summary[key]=Json::Int64(row[key].as<int64_t>());
                    for (auto key : {"first_loss","last_loss","best_loss","last_val_loss","average_speed"}) summary[key]=nullable(row,key);
                    summary["loss_change_percent"]=Json::Value(Json::nullValue);
                    if (!summary["first_loss"].isNull() && summary["first_loss"].asDouble()>0 && !summary["last_loss"].isNull())
                        summary["loss_change_percent"]=(summary["first_loss"].asDouble()-summary["last_loss"].asDouble())/summary["first_loss"].asDouble()*100;
                    body["latest_diagnostic"]=Json::Value(Json::nullValue);
                    if (!row["diagnostics"].isNull()) {
                        body["latest_diagnostic"]["global_step"]=Json::Int64(row["diagnostic_step"].as<int64_t>());
                        body["latest_diagnostic"]["diagnostics"]=slm_hardware::parseMetrics(row["diagnostics"].as<std::string>());
                    }
                    reply(drogon::HttpResponse::newHttpJsonResponse(body));
                }, [reply](const drogon::orm::DrogonDbException&) {
                    slm_hardware::sendError(reply,drogon::k503ServiceUnavailable,"Unable to retrieve monitor");
                },runId);
        }, {drogon::Get});

    // Keyset pagination leaves the original recent-metrics endpoint compatible.
    for (bool diagnosticsOnly : {false,true}) {
        const std::string route=diagnosticsOnly ? "/api/training-runs/{run_id}/diagnostics" : "/api/training-runs/{run_id}/metrics/history";
        drogon::app().registerHandler(route,
            [db,diagnosticsOnly](const drogon::HttpRequestPtr& request, Reply&& callback,const std::string& id) {
                auto reply=std::move(callback); int64_t runId,limit=1000,before=std::numeric_limits<int64_t>::max(),at=std::numeric_limits<int64_t>::max();
                bool valid=slm_hardware::parsePositiveInt64(id,runId);
                for (auto pair : {std::make_pair("limit",&limit), std::make_pair("before_step",&before),std::make_pair("at_step",&at)}) {
                    const auto value=request->getParameter(pair.first);
                    if (!value.empty()) valid=slm_hardware::parsePositiveInt64(value,*pair.second) && valid;
                }
                if (!valid || limit>1000) { slm_hardware::sendError(reply,drogon::k400BadRequest,"Invalid run ID, cursor or limit (1-1000)"); return; }
                const std::string sql="SELECT r.id AS run_id,m.* FROM training_runs r LEFT JOIN LATERAL ("
                    "SELECT global_step,epoch,train_loss,val_loss,learning_rate,tokens_per_second,diagnostics::text AS diagnostics "
                    "FROM training_metrics WHERE run_id=r.id AND global_step<$2 AND global_step<=$3 " +
                    std::string(diagnosticsOnly ? "AND diagnostics IS NOT NULL " : "") +
                    "ORDER BY global_step DESC LIMIT $4) m ON true WHERE r.id=$1 ORDER BY m.global_step DESC";
                db->execSqlAsync(sql,[reply,runId,limit](const drogon::orm::Result& result) {
                    if (result.empty()) { slm_hardware::sendError(reply,drogon::k404NotFound,"Run not found"); return; }
                    Json::Value body; body["status"]="ok";body["run_id"]=Json::Int64(runId);body["items"]=Json::Value(Json::arrayValue);
                    const bool empty=result[0]["global_step"].isNull();
                    const auto count=empty ? 0 : std::min<int64_t>(limit,result.size());
                    for (int64_t i=count-1;i>=0;--i) {
                        const auto row=result[i]; Json::Value item;
                        item["global_step"]=Json::Int64(row["global_step"].as<int64_t>()); item["epoch"]=row["epoch"].as<int>();
                        for (auto key : {"train_loss","val_loss","learning_rate","tokens_per_second"}) item[key]=nullable(row,key);
                        item["diagnostics"]=row["diagnostics"].isNull() ? Json::Value(Json::nullValue) : slm_hardware::parseMetrics(row["diagnostics"].as<std::string>());
                        body["items"].append(item);
                    }
                    body["count"]=Json::Int64(count);body["has_more"]=!empty && result.size()>static_cast<size_t>(limit);
                    body["next_before_step"]=count>0 ? body["items"][0]["global_step"] : Json::Value(Json::nullValue);
                    reply(drogon::HttpResponse::newHttpJsonResponse(body));
                },[reply](const drogon::orm::DrogonDbException&) {
                    slm_hardware::sendError(reply,drogon::k503ServiceUnavailable,"Unable to retrieve training history");
                },runId,before,at,limit+1);
            },{drogon::Get});
    }
}
} // namespace slm_monitor

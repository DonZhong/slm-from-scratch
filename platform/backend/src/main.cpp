#include <drogon/drogon.h>

#include <functional>

int main() {
    drogon::app().registerHandler(
        "/api/health",
        [](const drogon::HttpRequestPtr&,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            Json::Value body;
            body["status"] = "ok";
            body["service"] = "slm-platform-backend";

            auto response =
                drogon::HttpResponse::newHttpJsonResponse(body);

            callback(response);
        },
        {drogon::Get});

    drogon::app().addListener("0.0.0.0", 8080).run();

    return 0;
}
----------------------------------------------------------------------------------------------------
--
-- HTTP Post Event Sender
-- 通过 GameplayNotificationBus 事件触发 HttpRequestBus 请求
--
----------------------------------------------------------------------------------------------------

local HttpPostEventSender =
{
    Properties =
    {
        IncomingPostEventName = { default = "rundatacol" },
        DefaultMessage = { default = "Hello from Lua" },
        SceneName = { default = "" },
    },
}

local function EscapeJsonString(input)
    local s = tostring(input or "")
    s = s:gsub("\\", "\\\\")
    s = s:gsub("\"", "\\\"")
    s = s:gsub("\n", "\\n")
    s = s:gsub("\r", "\\r")
    s = s:gsub("\t", "\\t")
    return s
end

function HttpPostEventSender:OnActivate()
    Debug.Log("[HttpPostEventSender] OnActivate called.")
    self.postEventBus = nil
    if self.Properties.IncomingPostEventName ~= "" then
        local busId = GameplayNotificationId(self.entityId, self.Properties.IncomingPostEventName, "float")
        self.postEventBus = GameplayNotificationBus.Connect(self, busId)
        Debug.Log("[HttpPostEventSender] Listening event: " .. tostring(self.Properties.IncomingPostEventName))
    else
        Debug.Log("[HttpPostEventSender] IncomingPostEventName is empty, script will not receive events.")
    end
end

function HttpPostEventSender:OnDeactivate()
    Debug.Log("[HttpPostEventSender] OnDeactivate called.")
    if self.postEventBus then
        self.postEventBus:Disconnect()
        self.postEventBus = nil
    end
end

function HttpPostEventSender:OnEventBegin(value)
    self:SendHttpReport(value)
end

function HttpPostEventSender:OnEventUpdating(value)
end

function HttpPostEventSender:OnEventEnd(value)
end

local function ParseKeyValuePairs(text)
    local result = {}
    if text == nil then
        return result
    end

    for key, value in tostring(text):gmatch("([%w_%.%-]+)%s*:%s*([^%s]+)") do
        result[tostring(key)] = tostring(value)
    end
    return result
end

local function EncodeJsonContentObject(contentMap)
    local keys = {}
    for key, _ in pairs(contentMap) do
        table.insert(keys, key)
    end
    table.sort(keys)

    local parts = {}
    for _, key in ipairs(keys) do
        local value = tostring(contentMap[key] or "")
        table.insert(parts, string.format("\"%s\":\"%s\"", EscapeJsonString(key), EscapeJsonString(value)))
    end

    return "{" .. table.concat(parts, ",") .. "}"
end

function HttpPostEventSender:GetSceneName()
    return tostring(self.Properties.SceneName or "")
end

function HttpPostEventSender:BuildPayload(incomingValue)
    local text = tostring(incomingValue or "")
    if text == "" then
        text = tostring(self.Properties.DefaultMessage or "")
    end

    local contentMap = ParseKeyValuePairs(text)
    if next(contentMap) == nil then
        contentMap.message = text
    end

    contentMap.scenename = self:GetSceneName()
    contentMap.time = os.date("!%Y-%m-%dT%H:%M:%SZ")

    local contentJson = EncodeJsonContentObject(contentMap)
    return "{\"content\":" .. contentJson .. "}"
end

function HttpPostEventSender:SendHttpReport(incomingValue)
    Debug.Log("[HttpPostEventSender] SendHttpReport called. incomingValue=" .. tostring(incomingValue))
    if HttpRequestBus == nil or HttpRequestBus.Broadcast == nil then
        Debug.Log("[HttpPostEventSender] HttpRequestBus is not available in Lua BehaviorContext.")
        return
    end

    local payload = self:BuildPayload(incomingValue)
    HttpRequestBus.Broadcast.ReportSimData(payload)
    Debug.Log("[HttpPostEventSender] ReportSimData sent. payload=" .. payload)
end

return HttpPostEventSender

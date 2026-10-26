----------------------------------------------------------------------------------------------------
--
-- UI Text Controller Script
-- 用于通过Lua脚本控制UICanvas中Text元素的显示内容
-- 通过 GameplayNotificationBus 的 SetText 事件动态修改文本
--
----------------------------------------------------------------------------------------------------

local UITextController =
{
    Properties =
    {
        TextEntity = { default = EntityId() },
        ImageEntity = { default = EntityId() },
        DefaultText = { default = "" },
        IncomingTextEventName = { default = "SetText" },
        IncomingImageEnableEventName = { default = "SetImageEnabled" },
        DisplayDurationSeconds = { default = 0},
        BlinkFrequencyHz = { default = 0},
        FontSize = { default = 32.0 },
        TextColorR = { default = 1.0 },
        TextColorG = { default = 1.0 },
        TextColorB = { default = 1.0 },
        TextColorA = { default = 1.0 },
    },
}

function UITextController:GetTargetEntityId()
    if self.Properties.TextEntity:IsValid() then
        return self.Properties.TextEntity
    end
    return self.entityId
end

function UITextController:GetImageEntityId()
    if self.Properties.ImageEntity:IsValid() then
        return self.Properties.ImageEntity
    end
    return self.entityId
end

function UITextController:ApplyImageEnabledFromValue(value)
    local imageId = self:GetImageEntityId()
    if not imageId:IsValid() then
        return
    end

    local enabled = nil
    local valueType = type(value)
    if valueType == "boolean" then
        enabled = value
    elseif valueType == "number" then
        enabled = (value ~= 0)
    elseif valueType == "string" then
        local normalized = string.lower(string.gsub(string.gsub(value, "^%s+", ""), "%s+$", ""))
        if normalized == "1" or normalized == "true" or normalized == "on" or normalized == "enable" or normalized == "enabled" then
            enabled = true
        elseif normalized == "0" or normalized == "false" or normalized == "off" or normalized == "disable" or normalized == "disabled" then
            enabled = false
        end
    end

    if enabled ~= nil then
        UiElementBus.Event.SetIsEnabled(imageId, enabled)
    end
end

function UITextController:ApplyTextStyle()
    local targetId = self:GetTargetEntityId()
    local fontSize = self.Properties.FontSize or 0.0
    if fontSize > 0.0 then
        UiTextBus.Event.SetFontSize(targetId, fontSize)
    end

    local r = self.Properties.TextColorR or 1.0
    local g = self.Properties.TextColorG or 1.0
    local b = self.Properties.TextColorB or 1.0
    local a = self.Properties.TextColorA or 1.0
    UiTextBus.Event.SetColor(targetId, Color(r, g, b, a))
end

function UITextController:ParseHexColor(colorText)
    if type(colorText) ~= "string" then
        return nil
    end

    local hex = string.lower(colorText)
    hex = string.gsub(hex, "^%s+", "")
    hex = string.gsub(hex, "%s+$", "")
    hex = string.gsub(hex, "^0x", "")
    hex = string.gsub(hex, "^#", "")

    if hex == "" then
        return nil
    end

    local value = tonumber(hex, 16)
    if not value then
        return nil
    end

    if string.len(hex) == 6 then
        local r = math.floor(value / 0x10000) % 0x100
        local g = math.floor(value / 0x100) % 0x100
        local b = value % 0x100
        return Color(r / 255.0, g / 255.0, b / 255.0, 1.0)
    end

    if string.len(hex) == 8 then
        -- 8位格式按 AARRGGBB 解析
        local a = math.floor(value / 0x1000000) % 0x100
        local r = math.floor(value / 0x10000) % 0x100
        local g = math.floor(value / 0x100) % 0x100
        local b = value % 0x100
        return Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    end

    return nil
end

function UITextController:ParseIncomingPayload(rawText)
    local payload = {}
    if type(rawText) ~= "string" then
        return payload
    end

    for key, value in string.gmatch(rawText, "([%a_]+)%s*:%s*\"(.-)\"") do
        payload[string.lower(key)] = value
    end

    return payload
end

function UITextController:OnActivate()
    local targetId = self:GetTargetEntityId()
    self.hideTimerRemaining = -1.0
    self.blinkTimerRemaining = 0.0
    self.isBlinkVisible = true
    self.currentBlinkFrequencyHz = self.Properties.BlinkFrequencyHz or 0.0

    if self.Properties.DefaultText ~= "" then
        UiTextBus.Event.SetText(targetId, self.Properties.DefaultText)
    end
    self:ApplyTextStyle()
    --UiElementBus.Event.SetIsEnabled(targetId, true)

    self.textEventBus = nil
    if self.Properties.IncomingTextEventName ~= "" then
        local busId = GameplayNotificationId(self.entityId, self.Properties.IncomingTextEventName, "float")
        self.textEventBus = GameplayNotificationBus.Connect(self, busId)
    end

    self.imageEnableEventBus = nil
    self.imageEnableEventHandler = nil
    if self.Properties.IncomingImageEnableEventName ~= "" and self:GetTargetEntityId():IsValid() then
        local imageEventBusId = GameplayNotificationId(self.entityId, self.Properties.IncomingImageEnableEventName, "float")
      --  Debug.Log("[UITextController] OnActivate: 图片事件总线ID=" .. tostring(imageEventBusId))
        self.imageEnableEventHandler =
        {
            controller = self
        }
        function self.imageEnableEventHandler:OnEventBegin(value)
           -- Debug.Log("[UITextController] OnEventBegin: 图片事件总线ID=" .. tostring(value))
            self.controller:ApplyImageEnabledFromValue(value)
        end
        function self.imageEnableEventHandler:OnEventUpdating(value)
            --Debug.Log("[UITextController] OnEventUpdating: 图片事件总线ID=" .. tostring(value))
            self.controller:ApplyImageEnabledFromValue(value)
        end
        function self.imageEnableEventHandler:OnEventEnd(value)
        end
        self.imageEnableEventBus = GameplayNotificationBus.Connect(self.imageEnableEventHandler, imageEventBusId)
       -- UiElementBus.Event.SetIsEnabled(targetId, false)
    end

    self.tickBusHandler = TickBus.Connect(self)
    self.tickCount = 0

   -- Debug.Log("[UITextController] OnActivate: 脚本已激活, 目标实体=" .. tostring(targetId))
end

function UITextController:OnDeactivate()
    local targetId = self:GetTargetEntityId()
    UiElementBus.Event.SetIsEnabled(targetId, true)

    if self.tickBusHandler then
        self.tickBusHandler:Disconnect()
        self.tickBusHandler = nil
    end
    if self.textEventBus then
        self.textEventBus:Disconnect()
        self.textEventBus = nil
    end
    if self.imageEnableEventBus then
        self.imageEnableEventBus:Disconnect()
        self.imageEnableEventBus = nil
    end
    self.imageEnableEventHandler = nil
end

function UITextController:OnEventBegin(value)
   -- Debug.Log("[UITextController] OnEventBegin: " .. tostring(value))
    self:ApplyIncomingText(tostring(value))
end

function UITextController:OnEventUpdating(value)
    self:ApplyIncomingText(tostring(value))
end

function UITextController:OnEventEnd(value)
end

function UITextController:ApplyIncomingText(text)
    local targetId = self:GetTargetEntityId()
    local payload = self:ParseIncomingPayload(text)

    local hasMessageField = (payload.message ~= nil)
    local displayText = payload.message
    if not hasMessageField then
        displayText = text
    end

    UiTextBus.Event.SetText(targetId, displayText)

    local fontSize = tonumber(payload.size)
    if fontSize and fontSize > 0.0 then
        UiTextBus.Event.SetFontSize(targetId, fontSize)
    else
        local defaultFontSize = self.Properties.FontSize or 0.0
        if defaultFontSize > 0.0 then
            UiTextBus.Event.SetFontSize(targetId, defaultFontSize)
        end
    end

    local parsedColor = self:ParseHexColor(payload.color)
    if parsedColor then
        UiTextBus.Event.SetColor(targetId, parsedColor)
    else
        self:ApplyTextStyle()
    end

    UiElementBus.Event.SetIsEnabled(targetId, true)
    self.isBlinkVisible = true
    self.blinkTimerRemaining = 0.0

    local displayDuration = tonumber(payload.showtime)
    if displayDuration == nil then
        displayDuration = self.Properties.DisplayDurationSeconds or 0.0
    end
    if displayDuration > 0.0 then
        self.hideTimerRemaining = displayDuration
    else
        self.hideTimerRemaining = -1.0
    end

    local blinkFrequency = tonumber(payload.blinkfreq)
    if blinkFrequency == nil then
        blinkFrequency = self.Properties.BlinkFrequencyHz or 0.0
    end
    self.currentBlinkFrequencyHz = blinkFrequency
end

function UITextController:OnTick(deltaTime, timePoint)
    self.tickCount = self.tickCount + 1
    if self.tickCount == 1 then
        local targetId = self:GetTargetEntityId()
        if self.Properties.DefaultText ~= "" then
            UiTextBus.Event.SetText(targetId, self.Properties.DefaultText)
           -- Debug.Log("[UITextController] OnTick: 首帧设置默认文本 -> " .. self.Properties.DefaultText)
        end
    end

    if self.hideTimerRemaining > 0.0 then
        self.hideTimerRemaining = self.hideTimerRemaining - deltaTime
        if self.hideTimerRemaining <= 0.0 then
            local targetId = self:GetTargetEntityId()
            UiTextBus.Event.SetText(targetId, "")
            UiElementBus.Event.SetIsEnabled(targetId, true)
            self.hideTimerRemaining = -1.0
            --Debug.Log("[UITextController] Timed display expired, text cleared")
        end
    end

    local blinkFrequency = self.currentBlinkFrequencyHz or 0.0
    local targetId = self:GetTargetEntityId()
    if blinkFrequency > 0.0 then
        local currentText = UiTextBus.Event.GetText(targetId)
        if currentText ~= nil and currentText ~= "" then
            local toggleInterval = 0.5 / blinkFrequency
            self.blinkTimerRemaining = self.blinkTimerRemaining - deltaTime
            if self.blinkTimerRemaining <= 0.0 then
                self.isBlinkVisible = not self.isBlinkVisible
                UiElementBus.Event.SetIsEnabled(targetId, self.isBlinkVisible)
                self.blinkTimerRemaining = toggleInterval
            end
        else
            if not self.isBlinkVisible then
                self.isBlinkVisible = true
                UiElementBus.Event.SetIsEnabled(targetId, true)
            end
            self.blinkTimerRemaining = 0.0
        end
    else
        if not self.isBlinkVisible then
            self.isBlinkVisible = true
            UiElementBus.Event.SetIsEnabled(targetId, true)
        end
        self.blinkTimerRemaining = 0.0
    end
end

return UITextController

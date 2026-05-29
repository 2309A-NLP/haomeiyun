# 自定义角色API使用文档

## 概述

本文档介绍如何使用自定义角色功能，允许用户创建、管理和使用自己的法律咨询角色。

## 功能特性

- 创建自定义法律咨询角色
- 设置角色的专业领域、回答风格
- 自定义提示词模板
- 公开分享角色或私有使用
- 评分和评论角色
- 查看热门角色

## API接口

### 1. 创建自定义角色

**接口：** `POST /api/v1/roles/custom`

**请求参数：**

```json
{
  "name": "my_lawyer",
  "display_name": "我的专属律师",
  "description": "专注于婚姻家庭和财产纠纷的专业律师",
  "specialties": ["婚姻法", "财产分割", "继承法"],
  "answer_style": "professional",
  "tags": ["婚姻", "家庭", "财产"],
  "is_public": true,
  "avatar": "https://example.com/avatar.jpg",
  "system_prompt": "你是一位经验丰富的婚姻家庭律师...",
  "prompt_template": "专业领域：{specialties}

{context}

用户问题：{question}"
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | string | 是 | 角色唯一标识（英文） |
| display_name | string | 是 | 角色显示名称 |
| description | string | 是 | 角色描述（10-500字符） |
| specialties | array | 是 | 专业领域列表（1-10项） |
| answer_style | string | 否 | 回答风格：formal, friendly, professional, concise, detailed |
| tags | array | 否 | 标签列表 |
| is_public | boolean | 否 | 是否公开（默认false） |
| avatar | string | 否 | 头像URL |
| system_prompt | string | 否 | 系统级提示词 |
| prompt_template | string | 否 | 提示词模板 |

**回答风格说明：**

- `formal`: 正式严谨
- `friendly`: 友好亲切
- `professional`: 专业权威
- `concise`: 简洁明了
- `detailed`: 详细全面

**示例请求：**

```bash
curl -X POST "http://localhost:8000/api/v1/roles/custom?user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "family_lawyer",
    "display_name": "婚姻家事律师",
    "description": "专注婚姻家庭法律事务的专业律师",
    "specialties": ["婚姻法", "继承法", "财产分割"],
    "answer_style": "professional",
    "tags": ["婚姻", "家庭"],
    "is_public": true
  }'
```

### 2. 获取我的自定义角色列表

**接口：** `GET /api/v1/roles/custom/my`

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | integer | 是 | 用户ID |
| skip | integer | 否 | 跳过记录数（默认0） |
| limit | integer | 否 | 返回记录数（默认20） |

**示例请求：**

```bash
curl -X GET "http://localhost:8000/api/v1/roles/custom/my?user_id=1&skip=0&limit=10"
```

### 3. 获取公开的自定义角色列表

**接口：** `GET /api/v1/roles/custom/public`

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| skip | integer | 否 | 跳过记录数（默认0） |
| limit | integer | 否 | 返回记录数（默认20） |
| tag | string | 否 | 按标签筛选 |
| sort_by | string | 否 | 排序字段：created_at, rating, usage（默认created_at） |

**示例请求：**

```bash
# 按评分排序
curl -X GET "http://localhost:8000/api/v1/roles/custom/public?sort_by=rating"

# 按标签筛选
curl -X GET "http://localhost:8000/api/v1/roles/custom/public?tag=婚姻"
```

### 4. 获取自定义角色详情

**接口：** `GET /api/v1/roles/custom/{role_id}`

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| role_id | integer | 是 | 角色ID |
| user_id | integer | 否 | 用户ID（用于权限检查） |

**示例请求：**

```bash
curl -X GET "http://localhost:8000/api/v1/roles/custom/1?user_id=1"
```

### 5. 更新自定义角色

**接口：** `PUT /api/v1/roles/custom/{role_id}`

**请求参数：**

所有字段都是可选的，只更新提供的字段。

**示例请求：**

```bash
curl -X PUT "http://localhost:8000/api/v1/roles/custom/1?user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "更新后的名称",
    "description": "更新后的描述"
  }'
```

### 6. 删除自定义角色

**接口：** `DELETE /api/v1/roles/custom/{role_id}`

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| role_id | integer | 是 | 角色ID |
| user_id | integer | 是 | 用户ID |

**示例请求：**

```bash
curl -X DELETE "http://localhost:8000/api/v1/roles/custom/1?user_id=1"
```

### 7. 评分角色

**接口：** `POST /api/v1/roles/custom/{role_id}/rate`

**请求参数：**

```json
{
  "rating": 5,
  "comment": "这个角色回答很专业！"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| rating | integer | 是 | 评分1-5 |
| comment | string | 否 | 评论（最多200字符） |

**示例请求：**

```bash
curl -X POST "http://localhost:8000/api/v1/roles/custom/1/rate?user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 5,
    "comment": "非常专业的回答"
  }'
```

### 8. 使用角色（记录使用次数）

**接口：** `POST /api/v1/roles/custom/{role_id}/use`

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| role_id | integer | 是 | 角色ID |
| user_id | integer | 是 | 用户ID |

**示例请求：**

```bash
curl -X POST "http://localhost:8000/api/v1/roles/custom/1/use?user_id=1"
```

## 使用自定义角色进行对话

创建自定义角色后，可以在对话接口中使用该角色：

**接口：** `POST /api/v1/chat`

**请求参数：**

```json
{
  "message": "我想咨询离婚财产分割问题",
  "user_id": "1",
  "session_id": "session_123",
  "role_id": "1",
  "legal_field": "family",
  "stream": false
}
```

**注意：**
- `role_id` 为数字时，系统会查找自定义角色
- `role_id` 为字符串时（如 "lawyer"），使用系统默认角色

## 前端集成示例

### 创建角色表单

```javascript
async function createCustomRole(roleData) {
  const response = await fetch('/api/v1/roles/custom?user_id=1', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: roleData.name,
      display_name: roleData.displayName,
      description: roleData.description,
      specialties: roleData.specialties,
      answer_style: roleData.answerStyle,
      tags: roleData.tags,
      is_public: roleData.isPublic,
      avatar: roleData.avatar,
      system_prompt: roleData.systemPrompt,
      prompt_template: roleData.promptTemplate
    })
  });
  return response.json();
}
```

### 使用自定义角色

```javascript
async function chatWithCustomRole(message, roleId, userId) {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      user_id: userId,
      role_id: roleId,  // 使用自定义角色ID
      stream: true
    })
  });

  // 处理流式响应
  const reader = response.body.getReader();
  // ... 流式处理逻辑
}
```

## 注意事项

1. **角色名称唯一性**：每个用户的角色名称必须唯一
2. **权限控制**：只能编辑和删除自己创建的角色
3. **公开角色**：公开的角色可以被其他用户查看和使用
4. **评分系统**：用户可以对公开角色进行评分
5. **提示词模板**：如果不提供，系统会根据回答风格自动生成

## 最佳实践

1. **专业领域**：设置1-5个最相关的专业领域
2. **回答风格**：根据目标用户群体选择合适的风格
3. **提示词优化**：通过实际使用不断优化提示词
4. **标签使用**：使用准确的标签便于其他用户发现
5. **头像选择**：使用专业的头像增强可信度

## 常见问题

### Q: 如何修改已创建的角色？

A: 使用 `PUT /api/v1/roles/custom/{role_id}` 接口，只提供需要修改的字段。

### Q: 自定义角色和系统角色有什么区别？

A: 自定义角色由用户创建，可以完全自定义；系统角色由系统预设，提供标准的专业角色。

### Q: 如何让我的角色被更多人使用？

A: 
1. 将角色设置为公开（is_public=true）
2. 添加准确的标签
3. 优化提示词提高回答质量
4. 获得用户好评提升评分

### Q: 提示词模板中可以使用哪些变量？

A: 
- `{specialties}`: 专业领域
- `{context}`: 对话上下文
- `{question}`: 用户问题

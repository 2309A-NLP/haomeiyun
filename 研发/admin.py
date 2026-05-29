# admin.py
import requests
import os

BASE_URL = "http://localhost:8000"


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 60)
    print("      Legal RAG 系统管理工具")
    print("=" * 60)
    print("  1. 初始化角色数据")
    print("  2. 查看所有角色")
    print("  3. 查看系统信息")
    print("  4. 测试对话功能")
    print("  5. 查看知识库文档")
    print("  6. 查看服务状态")
    print("  0. 退出")
    print("=" * 60)


def check_system():
    """检查系统是否运行"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return True
    except:
        pass
    return False


def init_roles():
    """初始化角色"""
    print("\n🔄 正在初始化角色...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/roles/init-defaults")
        if response.status_code == 200:
            print("✅ 角色初始化成功！")
            data = response.json()
            print(f"   消息: {data.get('message', '')}")
        else:
            print(f"❌ 初始化失败: {response.status_code}")
            print(f"   错误: {response.text}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("   请确保系统已启动: python run.py")


def list_roles():
    """查看所有角色"""
    print("\n📋 正在获取角色列表...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/roles")
        if response.status_code == 200:
            roles = response.json()
            if roles:
                print(f"✅ 共有 {len(roles)} 个角色：\n")
                for i, role in enumerate(roles, 1):
                    print(f"{i}. {role['display_name']} ({role['name']})")
                    print(f"   描述: {role.get('description', '无')}")
                    if role.get('specialties'):
                        print(f"   专业领域: {', '.join(role['specialties'])}")
                    print()
            else:
                print("⚠️  暂无角色，请先执行初始化 (选项 1)")
        else:
            print(f"❌ 获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("   请确保系统已启动: python run.py")


def system_info():
    """查看系统信息"""
    print("\n🔍 正在获取系统信息...")
    try:
        response = requests.get(f"{BASE_URL}/system/info")
        if response.status_code == 200:
            info = response.json()
            print("\n✅ 系统信息：\n")
            print(f"系统名称: {info['system']['name']}")
            print(f"版本: {info['system']['version']}")
            print(f"调试模式: {info['system']['debug']}")
            print(f"\nLLM 配置:")
            print(f"  提供商: {info['llm']['provider']}")
            print(f"  模型: {info['llm']['model']}")
            print(f"\nRAG 配置:")
            print(f"  分块大小: {info['rag']['chunk_size']}")
            print(f"  检索数量: {info['rag']['top_k']}")
            print(f"\n数据库状态:")
            for db, status in info['databases'].items():
                status_icon = "✅" if "connected" in status else "❌"
                print(f"  {status_icon} {db}: {status}")
        else:
            print(f"❌ 获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")


def test_chat():
    """测试对话"""
    print("\n💬 测试对话功能")
    print("-" * 40)

    try:
        # 先获取角色
        response = requests.get(f"{BASE_URL}/api/v1/roles")
        if response.status_code != 200:
            print("❌ 无法获取角色列表")
            return

        roles = response.json()
        if not roles:
            print("⚠️  没有可用角色，请先初始化 (选项 1)")
            return

        print("\n可用角色：")
        for i, role in enumerate(roles, 1):
            print(f"  {i}. {role['display_name']}")

        # 选择角色
        while True:
            try:
                choice = input(f"\n请选择角色 (1-{len(roles)})，直接回车使用默认: ").strip()
                if not choice:
                    selected_role = roles[0]
                    break
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(roles):
                    selected_role = roles[choice_idx]
                    break
                else:
                    print(f"请输入 1-{len(roles)} 之间的数字")
            except ValueError:
                print("请输入有效数字")

        # 输入问题
        default_question = "我被公司开除了，能要求赔偿吗？"
        question = input(f"\n请输入问题（直接回车使用默认问题）: ").strip()
        if not question:
            question = default_question

        print(f"\n角色: {selected_role['display_name']}")
        print(f"问题: {question}")
        print("\n🤔 AI 正在思考...")

        # 发送请求
        chat_request = {
            "message": question,
            "user_id": "test_user",
            "role_id": selected_role['name']
        }

        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=chat_request,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 60)
            print("📝 回答：")
            print("=" * 60)
            print(result['reply'])
            print("\n" + "=" * 60)
            print(f"⚠️  风险等级: {result['risk_level']}")
            print(f"📚 引用来源: {len(result['citations'])} 条")
            print(f"⏱️  响应时间: {result.get('response_time', 'N/A')} 秒")
        else:
            print(f"❌ 对话失败: {response.status_code}")
            print(f"   错误: {response.text[:200]}")

    except Exception as e:
        print(f"❌ 错误: {e}")


def list_documents():
    """查看知识库文档"""
    print("\n📚 正在获取知识库文档...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/knowledge/documents")
        if response.status_code == 200:
            docs = response.json()
            if docs:
                print(f"✅ 共有 {len(docs)} 个文档：\n")
                for i, doc in enumerate(docs, 1):
                    print(f"{i}. {doc['title']}")
                    print(f"   类型: {doc['doc_type']}")
                    print(f"   领域: {doc['legal_field']}")
                    print(f"   状态: {doc['status']}")
                    print(f"   上传时间: {doc['created_at'][:19]}")
                    print()
            else:
                print("📭 暂无文档，请先上传")
        else:
            print(f"❌ 获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")


def show_status():
    """显示服务状态"""
    print("\n🔍 检查服务状态...")

    # 检查主服务
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ 主服务: 运行正常")
        else:
            print("⚠️  主服务: 状态异常")
    except:
        print("❌ 主服务: 未响应")

    # 检查 API
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API: {data.get('name', 'Legal RAG System')} v{data.get('version', 'unknown')}")
    except:
        print("❌ API: 无法访问")

    # 检查文档
    print(f"📚 API文档: {BASE_URL}/docs")


def main():
    """主函数"""
    clear_screen()
    print("\n🚀 Legal RAG 系统管理工具")
    print("\n提示：请确保系统已在另一个终端启动 (python run.py)\n")

    if not check_system():
        print("⚠️  警告：系统未运行或无法连接")
        print("   请在另一个终端执行: python run.py")
        print("   然后重新运行本工具\n")

        start = input("是否仍要继续？(y/n): ").strip().lower()
        if start != 'y':
            return

    while True:
        print_menu()
        choice = input("\n请选择操作 (0-7): ").strip()

        if choice == "1":
            init_roles()
        elif choice == "2":
            list_roles()
        elif choice == "3":
            system_info()
        elif choice == "4":
            test_chat()
        elif choice == "5":
            list_documents()
        elif choice == "6":
            show_status()
        elif choice == "0":
            print("\n👋 再见！")
            break
        else:
            print("❌ 无效选择，请输入 0-6 之间的数字")

        input("\n按回车键继续...")
        clear_screen()


if __name__ == "__main__":
    main()

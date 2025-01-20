import os  # Для работы с файловой системой и переменными окружения
import logging  # Для логирования действий агентов
from dotenv import load_dotenv  # Для загрузки переменных окружения из .env файла
from langchain_gigachat import GigaChat, GigaChatEmbeddings  # Для работы с GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_community.vectorstores import FAISS  # Для создания векторного хранилища
from langchain.text_splitter import CharacterTextSplitter  # Для разделения текста на части
import yaml  # Для работы с YAML файлами


# --- Загрузка API-ключа ---
load_dotenv()  # Загружаем переменные окружения из .env файла
# --- Загрузка API-ключа ---
load_dotenv()  # Загружаем переменные окружения из .env файла
gigachat_api_key = os.getenv("GIGACHAT_API_KEY")  # Получаем API-ключ GigaChat

# --- Инициализация GigaChat ---
giga = GigaChat(
    credentials=gigachat_api_key,  # Передаём API-ключ для авторизации
    scope="GIGACHAT_API_PERS",     # Указываем область применения API (например, персональный доступ)
    model="GigaChat",              # Задаём модель GigaChat для использования
    streaming=False,               # Отключаем потоковую передачу данных (ответ будет получен целиком)
    verify_ssl_certs=False         # Игнорируем проверку SSL-сертификатов (не рекомендуется в продакшене)
)

# --- Логирование ---
def setup_logger(agent_name):
    """
    Настраивает логгер для записи действий агента.
    :param agent_name: Имя агента для логирования.
    :return: Настроенный логгер.
    """
    log_folder = "logs"  # Папка для логов
    os.makedirs(log_folder, exist_ok=True)  # Создаём папку, если её нет
    logger = logging.getLogger(agent_name)  # Инициализируем логгер
    logger.setLevel(logging.INFO)  # Устанавливаем уровень логирования INFO
    log_file_path = os.path.join(log_folder, f"{agent_name}.log")  # Путь к лог-файлу
    handler = logging.FileHandler(log_file_path)  # Создаём обработчик для записи в файл
    formatter = logging.Formatter('%(asctime)s - %(message)s')  # Форматирование логов
    handler.setFormatter(formatter)  # Применяем формат к обработчику
    logger.addHandler(handler)  # Добавляем обработчик в логгер
    return logger  # Возвращаем логгер

# --- Агент коммуникации ---
def communication_agent(user_id, query):
    """
    Агент коммуникации:
    - Принимает запросы пользователей.
    - Передаёт запросы агенту-оркестратору.
    """
    logger = setup_logger("CommunicationAgent")  # Логирование для агента коммуникации
    logger.info(f"Получен запрос от пользователя {user_id}: {query}")  # Логируем запрос

    # Формируем данные для передачи агенту-оркестратору
    communication_data = {"user_id": user_id, "query": query}

    # Вызываем агент-оркестратор и передаём данные
    orchestrator_result = orchestrator_agent(communication_data, object_retriever)
    logger.info(f"Результат от агента-оркестратора: {orchestrator_result}")  # Логируем результат
    return orchestrator_result  # Возвращаем результат


# --- Агент-оркестратор ---
def orchestrator_agent(data, object_retriever):
    print("\n--- Агент-оркестратор ---")
    """
    Агент-оркестратор:
    - Определяет, каким агентам направить запрос.
    :param data: Данные от CommunicationAgent.
    :param object_retriever: Ретривер для поиска объектов.
    """
    logger = setup_logger("OrchestratorAgent")  # Логирование для оркестратора
    logger.info(f"Получен запрос от CommunicationAgent: {data}")  # Логируем данные

    query = data["query"]  # Извлекаем запрос пользователя

    # Поиск объектов в справочнике через RAG
    object_results = object_retriever.similarity_search(query, k=3)  # Поиск релевантных объектов
    object_list = "\n".join([result.page_content for result in object_results])  # Формируем текст из результатов поиска

    # Проверка через LLM: упоминаются ли объекты из справочника?
    llm_prompt = (
        f"Справочник объектов:\n{object_list}\n\n"
        f"Запрос пользователя: {query}\n\n"
        f"Есть ли в запросе упоминания объектов из справочника? Ответь 'Да' или 'Нет'."
    )
    llm_response = giga.invoke(llm_prompt).content.strip()  # Получаем ответ от GigaChat
    logger.info(f"Ответ LLM: {llm_response}")  # Логируем ответ LLM

    # Маршрутизация на основе ответа LLM
    if llm_response == "Да":
        result = repository_agent(query, systems_data_vectorstore)  # Вызов Repository-агента
        print(f"Передано агенту по работе с Репозиторием: {result}")
    elif llm_response == "Нет":
        result = knowledge_agent(query)  # Вызов агента базы знаний
        print(f"Передано агенту по работе с Внутренней базой знаний: {result}")

    else:
        result = "Ошибка: Некорректный ответ от LLM."  # Ошибка маршрутизации

    logger.info(f"Результат маршрутизации: {result}")  # Логируем результат
    return result  # Возвращаем результат

# --- Repository Агент (Ищет в YAML-файлах) ---

# --- Чтение данных из YAML ---
def load_systems_data(file_path):
    """
    Загружает данные систем из YAML файла.
    :param file_path: Путь к файлу YAML.
    :return: Список текстовых представлений объектов.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    systems = data.get("systems", {})

    # Формируем текстовые представления для индексации
    text_data = [
        f"id: {system_id}\n" + yaml.dump(system_data, allow_unicode=True)
        for system_id, system_data in systems.items()
    ]
    return text_data

# --- Создание векторного хранилища ---
def create_vectorstore(data, gigachat_api_key):
    """
    Создаёт векторное хранилище на основе данных.
    :param data: Список текстов для индексации.
    :param openai_api_key: Ключ OpenAI API.
    :return: Векторное хранилище FAISS.
    """
    embeddings = GigaChatEmbeddings(
        credentials=gigachat_api_key,
        verify_ssl_certs=False
    )
    vectorstore = FAISS.from_texts(data, embeddings)
    return vectorstore

def repository_agent(query, systems_data_vectorstore):
    print("\n--- Агент по работе с репозиторием ---")
    """
     Агент для поиска информации по объектам в репозитории.
     :param query: Запрос пользователя.
     :param vectorstore: Векторное хранилище данных.
     :return: Ответ на основе релевантных данных.
     """
    logger = setup_logger("RepositoryAgent")  # Логирование для оркестратора
    logger.info(f"Получен запрос от Агента-оркестратора: {query}")  # Логируем данные
    # Поиск релевантных документов
    results = systems_data_vectorstore.similarity_search(query, k=10)  # k - количество результатов
    relevant_data = "\n\n".join([result.page_content for result in results])

    # Логирование сформированных релевантных данных
    # logger.info(f"Сформированные релевантные данные (relevant_data):\n{relevant_data}")

    # Формируем запрос для LLM
    llm_prompt = (
        f"На основе следующих данных:\n{relevant_data}\n\n"
        f"Найди информацию по объекту в запросе: {query}.\n\n"
        f"Если данных недостаточно, скажи: 'Данных недостаточно для ответа на запрос.'"
    )
    response = giga.invoke(llm_prompt).content.strip()
    return response

# --- Агент внутренней базы знаний ---
def knowledge_agent(query):
    print("\n--- Агент по работе с Внутренней базой знаний ---")
    """
    Агент для поиска информации во внутренней базе знаний.
    Если релевантных данных недостаточно, запрос передаётся агенту LLM.
    :param query: Запрос пользователя.
    :return: Ответ на основе данных из базы знаний или направление запроса другому агенту.
    """
    logger = setup_logger("KnowledgeAgent")  # Создаём логгер для агента
    logger.info(f"Получен запрос от агента коммуникации: {query}")  # Логируем запрос

    # Чтение данных из базы знаний (файл knowledge_base.txt)
    knowledge_file = "knowledge_base.txt"  # Имя файла базы знаний
    with open(knowledge_file, "r", encoding="utf-8") as file:  # Открываем файл
        knowledge_data = file.read().splitlines()  # Читаем строки и разбиваем их на список

    # Создание векторного хранилища
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)  # Разделение текста на части
    documents = text_splitter.split_text("\n".join(knowledge_data))  # Формируем документы из текста
    embeddings = GigaChatEmbeddings(credentials=gigachat_api_key, verify_ssl_certs=False)  # Создаём эмбеддинги для текста
    vectorstore = FAISS.from_texts(documents, embeddings)  # Создаём векторное хранилище
    retriever = vectorstore.as_retriever()  # Настраиваем поиск по векторному хранилищу

    # Поиск в базе знаний
    results = retriever.invoke(query)  # Выполняем поиск по запросу

    if results:  # Если результат найден
        relevant_data = "\n".join([result.page_content for result in results])  # Объединяем результаты поиска
        logger.info(f"Релевантные данные найдены: {relevant_data}")  # Логируем найденные данные

        # Передача вопроса и данных в LLM
        llm_prompt = (
            f"На основе следующих данных из внутренней базы знаний:\n"
            f"{relevant_data}\n\n"
            f"Ответь на вопрос: {query}.\n\n"
            f"Если данных недостаточно, скажи: 'Во внутренней базе знаний ответа на данный вопрос нет.'"
        )
        response = giga.invoke(llm_prompt).content.strip()  # Получаем ответ от LLM
        logger.info(f"Ответ от LLM: {response}")  # Логируем результат
    else:  # Если результат не найден
        logger.info("Релевантные данные не найдены. Перенаправление запроса к LLM агенту.")  # Логируем отсутствие данных
        response = "Во внутренней базе знаний ответа на данный вопрос нет."  # Устанавливаем константу

    # Принятие решения на основе ответа
    if response == "Во внутренней базе знаний ответа на данный вопрос нет.":
        logger.info("Перенаправление запроса к агенту LLM.")  # Логируем переход
        return llm_recommender_agent(query)  # Перенаправляем запрос к LLM
    else:
        logger.info("Возврат ответа агенту коммуникации.")  # Логируем возврат
        return response  # Возвращаем релевантный ответ

# --- LLM Recommender Agent ---
def llm_recommender_agent(query):
    print("\n--- Агент для уточнения информации в LLM ---")
    """
    Агент для генерации ответа с помощью LLM.
    :param query: Запрос пользователя.
    :return: Ответ, сгенерированный LLM.
    """
    logger = setup_logger("LLMRecommenderAgent")  # Создаём логгер для агента
    logger.info(f"Получен запрос: {query}")  # Логируем запрос
    response = giga.invoke(query).content.strip()  # Отправляем запрос в LLM и получаем ответ
    logger.info(f"Ответ: {response}")  # Логируем результат
    return response  # Возвращаем ответ


# --- Тестирование системы ---
if __name__ == "__main__":
    # Настройка справочника объектов, поиск по которым должен осуществляться поиск в Репозитории
    object_data = [
        "Компания",
        "Пользователь",
        "Система",
        "Интеграция"
    ]
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    object_retriever = create_vectorstore(object_data, gigachat_api_key)

    # Пути к файлам
    systems_file = "objects/systems_test.yaml"
    # Загрузка данных о системах из репозитория
    systems_data = load_systems_data(systems_file)
    print(f"Загружены данные по Автоматизированным системам для индексации: {len(systems_data)} объектов")
    # Создание векторного хранилища c данными Автоматизированных Систем
    systems_data_vectorstore = create_vectorstore(systems_data, gigachat_api_key)
    print("Векторное хранилище с данными Автоматизированных Систем успешно создано.")

# Тестовый запрос
user_id = "user_001"
user_query = "Система CRM"

print("\n--- Агент коммуникации ---")
communication_data = communication_agent(user_id, user_query)
print(f"Итоговый ответ: {communication_data}")

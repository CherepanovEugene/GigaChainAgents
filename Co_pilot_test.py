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
    orchestrator_result = orchestrator_agent(communication_data)
    logger.info(f"Результат от агента-оркестратора: {orchestrator_result}")  # Логируем результат
    return orchestrator_result  # Возвращаем результат


def orchestrator_agent(data):
    """
    Агент-оркестратор:
    - Определяет, содержится ли объект в предустановленном списке.
    - Маршрутизирует запрос либо в RepositoryAgent, либо в KnowledgeAgent.

    :param data: Данные от CommunicationAgent.
    :return: Результат маршрутизации.
    """
    print("\n--- Агент-оркестратор ---")
    logger = setup_logger("OrchestratorAgent")  # Логирование
    logger.info(f"Получен запрос от CommunicationAgent: {data}")  # Логируем данные

    query = data["query"]  # Извлекаем запрос пользователя

    #  СПИСОК ОБЪЕКТОВ ДЛЯ ПОИСКА
    object_list = [
        "Компания",
        "Пользователь",
        "Система",
        "Интеграция"
    ]  # Просто список доступных объектов

    # --- Проверяем через LLM, упоминается ли объект в запросе ---
    llm_prompt = (
        f"Дан список объектов:\n{', '.join(object_list)}\n\n"
        f"Запрос пользователя: {query}\n\n"
        f"Есть ли в запросе упоминание одного из объектов в списке? Ответь 'Да' или 'Нет'."
    )

    llm_response = giga.invoke(llm_prompt).content.strip()  # Получаем ответ от LLM
    logger.info(f"Ответ LLM: {llm_response}")  # Логируем ответ LLM

    # --- Маршрутизация запроса ---
    if llm_response == "Да":
        result = repository_agent(query)  # Направляем в RepositoryAgent
        print(f"Передано агенту по работе с Репозиторием: {result}")
    elif llm_response == "Нет":
        result = knowledge_agent(query)  # Направляем в KnowledgeAgent
        print(f"Передано агенту по работе с Внутренней базой знаний: {result}")
    else:
        result = "Ошибка: Некорректный ответ от LLM."  # Ошибка маршрутизации

    logger.info(f"Результат маршрутизации: {result}")  # Логируем результат
    return result  # Возвращаем результат


# --- Repository Агент (Ищет в YAML-файлах) ---

# --- Функция загрузки данных из YAML ---
def load_systems_data(yaml_file):
    """
    Загружает данные систем из YAML-файла.
    :param yaml_file: Путь к YAML-файлу.
    :return: Список текстовых представлений объектов.
    """
    logger = setup_logger("LoadSystemsData")
    logger.info(f"Загрузка данных из {yaml_file}")

    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)  # Читаем YAML-файл

    systems = data.get("systems", {})  # Достаём объекты "systems"

    # Формируем текстовые представления для индексации
    text_data = [
        f"id: {system_id}\n" + yaml.dump(system_data, allow_unicode=True)
        for system_id, system_data in systems.items()
    ]

    logger.info(f"Загружено {len(text_data)} объектов из {yaml_file}")
    return text_data


# --- Функция загрузки векторного хранилища ---
def load_vectorstore(vectorstore_path):
    """
    Загружает локальное векторное хранилище FAISS, если оно существует.
    :param vectorstore_path: Путь к файлу хранилища.
    :return: Объект FAISS или None, если файла нет.
    """
    logger = setup_logger("VectorStoreLoader")

    if os.path.exists(vectorstore_path):
        logger.info(f"Загрузка векторного хранилища из {vectorstore_path}...")
        return FAISS.load_local(vectorstore_path, embeddings = GigaChatEmbeddings(                  # Создаём эмбеддинги для текста
            credentials=gigachat_api_key,
            verify_ssl_certs=False
        ),
                                allow_dangerous_deserialization=True)

    logger.warning(f"Файл {vectorstore_path} не найден. Требуется создание нового хранилища.")
    return None  # Если файла нет, возвращаем None


# --- Функция создания векторного хранилища ---
def create_vectorstore(data, vectorstore_path):
    """
    Создаёт и сохраняет векторное хранилище FAISS.
    :param data: Список текстов для индексации.
    :param vectorstore_path: Путь для сохранения.
    :return: Объект FAISS.
    """
    logger = setup_logger("VectorStoreCreator")
    logger.info(f"Создание векторного хранилища в {vectorstore_path}")

    embeddings = GigaChatEmbeddings(
        credentials=gigachat_api_key,
        verify_ssl_certs=False
    )
    vectorstore = FAISS.from_texts(data, embeddings)

    # Сохраняем хранилище локально
    vectorstore.save_local(vectorstore_path)
    logger.info(f"Векторное хранилище сохранено в {vectorstore_path}")

    return vectorstore


# --- Функция агента репозитория ---
def repository_agent(query):
    """
    Агент для поиска информации по объектам в репозитории.
    :param query: Запрос пользователя.
    :return: Ответ LLM на основе данных.
    """
    global systems_data_vectorstore  # Используем глобальную переменную

    logger = setup_logger("RepositoryAgent")
    logger.info(f"Получен запрос: {query}")

    # 1. Загружаем векторное хранилище, если оно ещё не загружено
    if systems_data_vectorstore is None:
        logger.info("Векторное хранилище не загружено. Загружаем...")
        systems_data_vectorstore = load_vectorstore(vectorstore_path)

        if systems_data_vectorstore is None:  # Если хранилище отсутствует, создаём новое
            logger.info("Векторное хранилище не найдено. Создаём новое...")
            systems_data = load_systems_data(systems_file)  # Загружаем YAML-файл
            systems_data_vectorstore = create_vectorstore(systems_data, vectorstore_path)  # Создаём FAISS-хранилище

    # 2. Выполняем поиск по хранилищу
    results = systems_data_vectorstore.as_retriever().invoke(query)

    if results:
        relevant_data = "\n\n".join([result.page_content for result in results])
        logger.info(f"Найдено {len(results)} релевантных фрагментов:\n{relevant_data}")

        # 3. Формируем запрос к LLM
        llm_prompt = (
            f"На основе следующих данных:\n{relevant_data}\n\n"
            f"Ответь на запрос: {query}.\n\n"
            f"Если данных недостаточно, скажи: 'Данных недостаточно для ответа'."
        )
        response = giga.invoke(llm_prompt).content.strip()
    else:
        logger.warning("Данные не найдены.")
        response = "Данных недостаточно для ответа."

    logger.info(f"Ответ от RepositoryAgent: {response}")
    return response


def repository_agent(query):
    """
    Агент для поиска информации по объектам в YAML-репозитории.
    :param query: Запрос пользователя.
    :return: Ответ LLM на основе данных.
    """
    global systems_data_vectorstore  # Используем глобальную переменную

    logger = setup_logger("RepositoryAgent")
    logger.info(f"Получен запрос: {query}")

    # 1. Загружаем векторное хранилище, если оно ещё не загружено
    if systems_data_vectorstore is None:
        logger.info("Векторное хранилище не загружено. Загружаем...")
        systems_data_vectorstore = load_vectorstore(vectorstore_path)

        if systems_data_vectorstore is None:  # Если хранилище отсутствует, создаём новое
            logger.info("Векторное хранилище не найдено. Создаём новое...")
            systems_data = load_systems_data(systems_file)  # Загружаем YAML-файл
            systems_data_vectorstore = create_vectorstore(systems_data, vectorstore_path)  # Создаём FAISS-хранилище

    # 2. Выполняем поиск по хранилищу
    results = systems_data_vectorstore.as_retriever().invoke(query)

    if results:
        relevant_data = "\n\n".join([result.page_content for result in results])
        logger.info(f"Найдено {len(results)} релевантных фрагментов:\n{relevant_data}")

        # 3. Формируем запрос к LLM
        llm_prompt = (
            f"На основе следующих данных:\n{relevant_data}\n\n"
            f"Ответь на запрос: {query}.\n\n"
            f"Если данных недостаточно, скажи: 'Данных недостаточно для ответа'."
        )
        response = giga.invoke(llm_prompt).content.strip()
    else:
        logger.warning("Данные не найдены.")
        response = "Данных недостаточно для ответа."

    logger.info(f"Ответ от RepositoryAgent: {response}")
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
    # knowledge_file = "knowledge_base.txt"  # Имя файла базы знаний
    # with open(knowledge_file, "r", encoding="utf-8") as file:  # Открываем файл
    #     knowledge_data = file.read().splitlines()  # Читаем строки и разбиваем их на список
    # # Создание векторного хранилища
    # text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)  # Разделение текста на части
    # documents = text_splitter.split_text("\n".join(knowledge_data))  # Формируем документы из текста
    # embeddings = GigaChatEmbeddings(                  # Создаём эмбеддинги для текста
    #         credentials=gigachat_api_key,
    #         verify_ssl_certs=False
    #     )
    # knowledge_vectorstore = FAISS.from_texts(documents, embeddings)  # Создаём векторное хранилище

    # Путь для сохранения хранилища
    knowledge_vectorstore_path = "vectorstores/knowledge_vectorstore.faiss"
    # Сохранение FAISS-хранилища
    # knowledge_vectorstore.save_local(knowledge_vectorstore_path)

    # Загрузка FAISS-хранилища
    knowledge_vectorstore = FAISS.load_local(knowledge_vectorstore_path, embeddings = GigaChatEmbeddings(                  # Создаём эмбеддинги для текста
            credentials=gigachat_api_key,
            verify_ssl_certs=False
        ),
                                allow_dangerous_deserialization=True)
    retriever = knowledge_vectorstore.as_retriever()  # Настраиваем поиск по векторному хранилищу

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
    vectorstore_path = "vectorstores/systems_vectorstore.faiss"  # Путь к файлу хранилища
    systems_file = "objects/systems_test.yaml"  # Файл с объектами

    # Загружаем векторное хранилище или создаём его, если нет файла
    systems_data_vectorstore = load_vectorstore(vectorstore_path)
    if systems_data_vectorstore is None:  # Если хранилище отсутствует, создаём новое
        systems_data = load_systems_data(systems_file)
        systems_data_vectorstore = create_vectorstore(systems_data, vectorstore_path)


    # Тестовый запрос
    user_id = "user_001"
    user_query = "Нейронные сети"

    print("\n--- Агент коммуникации ---")
    communication_data = communication_agent(user_id, user_query)
    print(f"Итоговый ответ: {communication_data}")

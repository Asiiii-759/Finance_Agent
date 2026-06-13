from __future__ import annotations

import os
from pathlib import Path
import sys
import typing as t
from functools import cached_property
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource, SettingsConfigDict
from pydantic import BaseModel, ConfigDict

CHATCHAT_ROOT = Path(
    os.environ.get("FINANCE_RAG_ROOT")
    or os.environ.get("CHATCHAT_ROOT")
    or Path(__file__).parent
).resolve()

load_dotenv(CHATCHAT_ROOT / ".env")

class MyBaseModel(BaseModel):
    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="allow",
        env_file_encoding="utf-8",
    )
    
class BaseFileSettings(BaseSettings):
    model_config = SettingsConfigDict(
        use_attribute_docstrings=True,
        extra="ignore",
        yaml_file_encoding="utf-8",
        env_file_encoding="utf-8",
    )

    def model_post_init(self, __context: os.Any) -> None:
        self._auto_reload = True
        return super().model_post_init(__context)

    @property
    def auto_reload(self) -> bool:
        return self._auto_reload
    
    @auto_reload.setter
    def auto_reload(self, val: bool):
        self._auto_reload = val

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls)

    def create_template_file(
        self,
        model_obj: BaseFileSettings=None,
        dump_kwds: t.Dict={},
        sub_comments: t.Dict[str, SubModelComment]={},
        write_file: bool | str | Path = False,
        file_format: t.Literal["yaml", "json"] = "yaml",
    ) -> str:
        if model_obj is None:
            model_obj = self
        if file_format == "yaml":
            template = YamlTemplate(model_obj=model_obj, dump_kwds=dump_kwds, sub_comments=sub_comments)
            return template.create_yaml_template(write_to=write_file)
        else:
            dump_kwds.setdefault("indent", 4)
            data = model_obj.model_dump_json(**dump_kwds)
            if write_file:
                write_file = self.model_config.get("json_file")
                with open(write_file, "w", encoding="utf-8") as fp:
                    fp.write(data)
            return data


class BasicSettings(BaseFileSettings):
    """
    服务器基本配置信息
    除 log_verbose/HTTPX_DEFAULT_TIMEOUT 修改后即时生效
    其它配置项修改后都需要重启服务器才能生效，服务运行期间请勿修改
    """
    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "basic_settings.yaml")
    log_verbose: bool = False
    """是否开启日志详细信息"""
    HTTPX_DEFAULT_TIMEOUT: float = 300
    """httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。"""

    # @computed_field
    @cached_property
    def PACKAGE_ROOT(self) -> Path:
        """代码根目录"""
        return Path(__file__).parent

    # @computed_field
    @cached_property
    def DATA_PATH(self) -> Path:
        """用户数据根目录"""
        p = CHATCHAT_ROOT / "Data"
        return p

    # @computed_field
    @cached_property
    def LOG_PATH(self) -> Path:
        """日志存储路径"""
        p = self.DATA_PATH / "logs"
        return p

    # @computed_field
    @cached_property
    def BASE_TEMP_DIR(self) -> Path:
        """临时文件目录，主要用于文件对话"""
        p = self.DATA_PATH / "temp"
        (p / "openai_files").mkdir(parents=True, exist_ok=True)
        return p

    KB_ROOT_PATH: str = str(CHATCHAT_ROOT / "Data/knowledge_base")
    """知识库默认存储路径"""

    DB_ROOT_PATH: str = str(CHATCHAT_ROOT / "Data/knowledge_base/info.db")
    """数据库默认存储路径。如果使用sqlite，可以直接修改DB_ROOT_PATH；如果使用其它数据库，请直接修改SQLALCHEMY_DATABASE_URI。"""

    SQLALCHEMY_DATABASE_URI:str = "sqlite:///" + str(CHATCHAT_ROOT / "Data/knowledge_base/info.db")
    """知识库信息数据库连接URI"""

    DEFAULT_BIND_HOST: str = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
    """
    各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
    Windows 下 WEBUI 自动弹出浏览器时，如果地址为 "0.0.0.0" 是无法访问的，需要手动修改地址栏
    """
    
    paddle_model_url: str = "http://127.0.0.1:8118/v1"

    API_SERVER: dict = {"host": DEFAULT_BIND_HOST, "port": 7861, "public_host": "127.0.0.1", "public_port": 7861}
    """API 服务器地址。其中 public_host 用于生成云服务公网访问链接（如知识库文档链接）"""


    def make_dirs(self):
        '''创建所有数据目录'''
        for p in [
            self.DATA_PATH,
            self.LOG_PATH,
            self.BASE_TEMP_DIR,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        Path(self.KB_ROOT_PATH).mkdir(parents=True, exist_ok=True)


class KBSettings(BaseFileSettings):
    """知识库相关配置"""

    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "kb_settings.yaml")

    DEFAULT_KNOWLEDGE_BASE: str = "Finance"
    """默认使用的知识库"""

    INDEX_TYPE: str = "HNSW"
    """知识库中向量索引类型"""

    CHUNK_SIZE: int = 512
    """知识库中单段文本长度"""

    OVERLAP_SIZE: int = int(CHUNK_SIZE * 0.25)
    """知识库中相邻文本重合长度"""

    DEFAULT_VS_TYPE: t.Literal["faiss", "milvus"] = "faiss"
    """默认向量库/全文检索引擎类型"""

    DEFAULT_EXMPERIMENT: str = f"{DEFAULT_KNOWLEDGE_BASE}_{DEFAULT_VS_TYPE}_{CHUNK_SIZE}_{INDEX_TYPE}"
    """默认使用的实验名称（自动拼接）"""

    VECTOR_SEARCH_TOP_K: int = 10
    """知识库匹配向量数量"""

    SCORE_THRESHOLD: float = 0.5
    """知识库匹配相关度阈值"""

    KB_INFO: t.Dict[str, str] = {"Finance": "用于金融研报相关的知识问答"}
    """每个知识库的初始化介绍，用于在初始化知识库时显示和Agent调用，没写则没有介绍，不会被Agent调用。"""

    kbs_config: t.Dict[str, t.Dict] = {
            "faiss": {},
            "milvus": {
                "host": "127.0.0.1",
                "port": "19530",
                "user": "",
                "password": "",
                "secure": False
            },
        }
    """可选向量库类型及对应配置"""

    TEXT_SPLITTER_NAME: str = "RecursiveChineseBlockSplitter"
    """TEXT_SPLITTER 名称"""
    
    TOKENIZER_FILE: str = str(CHATCHAT_ROOT / "models/bge-m3/sentencepiece.bpe.model")

    EMBEDDING_KEYWORD_FILE: str = str(CHATCHAT_ROOT / "Data/knowledge_base/embedding_keywords.txt")
    """Embedding模型定制词语的词表文件"""


class PlatformConfig(MyBaseModel):
    """模型加载平台配置"""

    platform_name: str = "openai"
    """平台名称"""

    platform_type: t.Literal["xinference", "ollama", "oneapi", "fastchat", "openai", "custom openai"] = "openai"
    """平台类型"""

    api_base_url: str = "http://127.0.0.1:9997/v1"
    """openai api url"""

    api_key: str = "EMPTY"
    """api key if available"""

    api_proxy: str = ""
    """API 代理"""

    api_concurrencies: int = 5
    """该平台单模型最大并发数"""

    auto_detect_model: bool = False
    """是否自动获取平台可用模型列表。设为 True 时下方不同模型类型可自动检测"""

    llm_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的大语言模型列表，auto_detect_model 设为 True 时自动检测"""

    embed_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的嵌入模型列表，auto_detect_model 设为 True 时自动检测"""

    rerank_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的重排模型列表，auto_detect_model 设为 True 时自动检测"""


class ApiModelSettings(BaseFileSettings):
    """模型配置项"""

    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "model_settings.yaml")

    DEFAULT_LLM_MODEL: str = "qwen3"
    """默认选用的 LLM 名称"""

    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-v4"
    """默认选用的 Embedding 名称"""

    DEFAULT_EMBEDDING_MODEL_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    """默认 Embedding 模型的 API 地址"""

    DEFAULT_EMBEDDING_DIMENSIONS: int = 1024
    """默认 Embedding 维度。text-embedding-v4 支持 2048/1536/1024/768/512/256/128/64。"""
    
    DEFAULT_RERANKING_MODEL: str = "bge-reranker-v2-m3"
    """默认 Reranker 模型"""

    HISTORY_LEN: int = 3
    """默认历史对话轮数"""

    MAX_TOKENS: t.Optional[int] = None
    """大模型最长支持的长度，如果不填写，则使用模型默认的最大长度，如果填写，则为用户设定的最大长度"""

    TEMPERATURE: float = 0.7
    """LLM通用对话参数"""

    LLM_MODEL_CONFIG: t.Dict[str, t.Dict] = {
            "llm_model": {
                "model": "",
                "temperature": 0.9,
                "max_tokens": 4096,
                "history_len": 10,
                "prompt_name": "default",
                "callbacks": True,
            }
        }
    """
    LLM模型配置，包括了不同模态初始化参数。
    `model` 如果留空则自动使用 DEFAULT_LLM_MODEL
    """

    MODEL_PLATFORMS: t.List[PlatformConfig] = [
            PlatformConfig(**{
                "platform_name": "oneapi",
                "platform_type": "oneapi",
                "api_base_url": "http://127.0.0.1:3000/v1",
                "api_key": "EMPTY",
                "api_concurrencies": 5,
                "auto_detect_model": False,
                "llm_models": [
                    "qwen-turbo",
                    "qwen-plus",
                    "qwen-max",
                ],
                "embed_models": ["bge-m3"],
                "rerank_models": ["bge-reranker-v2-m3"],
            }),
        ]
    """模型平台配置"""


class ToolSettings(BaseFileSettings):
    """Agent 工具配置项"""
    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "tool_settings.yaml",
                                      extra="allow")

    search_local_knowledgebase: dict = {
        "use": False,
        "top_k": 5,
        "score_threshold": 0.5,
        "conclude_prompt": {
            "with_result": '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题"，'
            "不允许在答案中添加编造成分，答案请使用中文。 </指令>\n"
            "<已知信息>{{ context }}</已知信息>\n"
            "<问题>{{ question }}</问题>\n",
            "without_result": "请你根据我的提问回答我的问题:\n"
            "{{ question }}\n"
            "请注意，你必须在回答结束后强调，你的回答是根据你的经验回答而不是参考资料回答的。\n",
        },
    }
    '''本地知识库工具配置项'''

    search_internet: dict = {
        "use": False,
        "search_engine_name": "duckduckgo",
        "search_engine_config": {
            "bing": {
                "bing_search_url": "https://api.bing.microsoft.com/v7.0/search",
                "bing_key": "",
            },
            "metaphor": {
                "metaphor_api_key": "",
                "split_result": False,
                "chunk_size": 500,
                "chunk_overlap": 0,
            },
            "duckduckgo": {},
            "searx": {
                "host": "https://metasearx.com",
                "engines": [],
                "categories": [],
                "language": "zh-CN",
            }
        },
        "top_k": 5,
        "verbose": "Origin",
        "conclude_prompt": "<指令>这是搜索到的互联网信息，请你根据这些信息进行提取并有调理，简洁的回答问题。如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。 "
        "</指令>\n<已知信息>{{ context }}</已知信息>\n"
        "<问题>\n"
        "{{ question }}\n"
        "</问题>\n",
    }
    '''搜索引擎工具配置项。推荐自己部署 searx 搜索引擎，国内使用最方便。'''

    arxiv: dict = {
        "use": False,
    }

    weather_check: dict = {
        "use": False,
        "api_key": "",
    }
    '''心知天气（https://www.seniverse.com/）工具配置项'''

    search_youtube: dict = {
        "use": False,
    }

    wolfram: dict = {
        "use": False,
        "appid": "",
    }

    calculate: dict = {
        "use": False,
    }
    '''numexpr 数学计算工具配置项'''

    url_reader: dict = {
        "use": False,
        "timeout": "10000",
    }
    '''URL内容阅读（https://r.jina.ai/）工具配置项
    请确保部署的网络环境良好，以免造成超时等问题'''



class PromptSettings(BaseFileSettings):
    """Prompt 模板.除 Agent 模板使用 f-string 外，其它均使用 jinja2 格式"""
    
    llm_model: dict = {
        "default": "{{input}}",
        "with_history": (
            "以下是用户与AI之间的友好对话。\n"
            "AI会根据上下文提供详细、准确、有用的回答。\n"
            "如果AI不知道问题的答案，会如实说明，不会编造信息。\n\n"
            "当前对话历史：\n"
            "{{history}}\n"
            "用户：{{input}}\n"
            "AI："
        ),
    }
    '''普通 LLM 用模板'''

    rag: dict = {
        "default": (
            "【指令】根据已知信息，简洁和专业的来回答问题。"
            "如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。\n\n"
            "【已知信息】{{context}}\n\n"
            "【问题】{{question}}\n"
            ),
        "empty": (
            "请你回答我的问题:\n"
            "{{question}}"
        ),
    }
    '''RAG 用模板，可用于知识库问答、文件对话、搜索引擎对话'''


class SettingsContainer:
    CHATCHAT_ROOT = CHATCHAT_ROOT

    basic_settings: BasicSettings = BasicSettings()
    kb_settings: KBSettings = KBSettings()
    model_settings: ApiModelSettings = ApiModelSettings()
    tool_settings: ToolSettings = ToolSettings()
    prompt_settings: PromptSettings = PromptSettings()

    def createl_all_templates(self):
        self.basic_settings.create_template_file(write_file=True)
        self.kb_settings.create_template_file(write_file=True)
        self.model_settings.create_template_file(
            sub_comments={
                "MODEL_PLATFORMS": {"model_obj": PlatformConfig(), "is_entire_comment": True}
            },
            write_file=True
        )
        self.tool_settings.create_template_file(write_file=True, file_format="yaml")
        self.prompt_settings.create_template_file(write_file=True, file_format="yaml")

Settings = SettingsContainer()

if __name__ == "__main__":
    Settings.createl_all_templates()

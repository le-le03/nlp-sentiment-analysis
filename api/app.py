"""FastAPI 情感分析服务"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="中文情感分析系统 API",
    description="基于 CNN/LSTM/Hybrid 集成模型的中文文本情感分析服务，支持单条和批量预测",
    version="3.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型
model = None
tokenizer = None
scaler = None
vectorizer = None


class PredictRequest(BaseModel):
    """单条预测请求"""
    text: str = Field(..., description="待分析的中文文本", example="这家酒店服务非常热情，房间干净整洁")

    class Config:
        title = "单条预测请求"
        schema_extra = {
            "example": {
                "text": "这家酒店服务非常热情，房间干净整洁"
            }
        }


class BatchPredictRequest(BaseModel):
    """批量预测请求"""
    texts: List[str] = Field(..., description="待分析的文本列表（多条）", example=["服务很好", "房间很脏"])

    class Config:
        title = "批量预测请求"
        schema_extra = {
            "example": {
                "texts": ["服务很好", "房间很脏", "物超所值"]
            }
        }


class PredictResponse(BaseModel):
    """单条预测响应"""
    text: str = Field(..., description="输入的文本")
    sentiment: str = Field(..., description="情感判断结果：正面/负面")
    confidence: float = Field(..., description="预测置信度（0-1）")
    positive_prob: float = Field(..., description="正面情感概率")
    negative_prob: float = Field(..., description="负面情感概率")

    class Config:
        title = "单条预测响应"


class BatchPredictResponse(BaseModel):
    """批量预测响应"""
    results: List[PredictResponse] = Field(..., description="预测结果列表")
    total: int = Field(..., description="总条数")

    class Config:
        title = "批量预测响应"


@app.on_event("startup")
async def load_model():
    """启动时加载训练好的模型"""
    global model, tokenizer, scaler, vectorizer
    try:
        from predictor import SentimentPredictor
        model = SentimentPredictor()
        logger.info("模型加载成功")
    except Exception as e:
        logger.warning(f"模型加载失败: {e}，请先运行训练脚本 python src/main.py")
        model = None


@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务是否正常运行，以及模型是否加载成功"""
    return {
        "status": "服务正常",
        "模型已加载": model is not None,
        "版本": "3.0.0"
    }


@app.get("/model/info", summary="模型信息")
async def model_info():
    """查看当前使用的模型信息"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请先运行训练脚本")
    return {
        "模型类型": "集成模型 (Ensemble)",
        "版本": "3.0.0",
        "包含模型": ["CNN", "LSTM", "Hybrid (CNN+LSTM)"],
        "训练数据集": "ChnSentiCorp 中文酒店评论"
    }


@app.post("/predict", response_model=PredictResponse, summary="单条文本情感预测")
async def predict(request: PredictRequest):
    """输入一段中文文本，返回情感分析结果（正面/负面）和置信度"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请先运行训练脚本")

    try:
        result = model.predict(request.text)
        return PredictResponse(
            text=request.text,
            sentiment=result.get("sentiment", "未知"),
            confidence=result.get("confidence", 0.0),
            positive_prob=result.get("positive_prob", 0.0),
            negative_prob=result.get("negative_prob", 0.0)
        )
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="批量文本情感预测")
async def batch_predict(request: BatchPredictRequest):
    """输入多段中文文本，批量返回情感分析结果"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请先运行训练脚本")

    try:
        results = []
        for text in request.texts:
            result = model.predict(text)
            results.append(PredictResponse(
                text=text,
                sentiment=result.get("sentiment", "未知"),
                confidence=result.get("confidence", 0.0),
                positive_prob=result.get("positive_prob", 0.0),
                negative_prob=result.get("negative_prob", 0.0)
            ))
        return BatchPredictResponse(results=results, total=len(results))
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

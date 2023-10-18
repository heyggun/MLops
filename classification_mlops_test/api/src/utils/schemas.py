from pydantic import BaseModel
from typing import Optional, List


class UserData(BaseModel):
    autoNo: int
    memNo: int
    mateConts: Optional[str]
    familyConts: Optional[str]


class UserInfo(BaseModel):
    UserInfo: List[dict]


class UserOut(BaseModel):
    autoNo: int
    memNo: int
    matePred: List[int]
    familyPred: List[int]
    reqTime: float

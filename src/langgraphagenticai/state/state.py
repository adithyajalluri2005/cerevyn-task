from typing_extensions import TypedDict,List
from typing import Annotated
from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional
class NLUOutput(BaseModel):
    intent: Literal["Billing Issue", "SIM Not Working", "No Network Coverage", "Internet Speed Slow", "Data Not Working After Recharge", "Call Drops Frequently"] = Field(description="Classified intent from predefined categories.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    entities: Dict[str, str] = Field(default_factory=dict, description="Extracted entities.")

class TranscriptEntry(TypedDict):
    text: str
    ts: float
    speaker: Literal['user', 'agent']

class CallState(TypedDict):
    call_id: str
    transcript: List[TranscriptEntry]
    clean_text: str
    intent: str
    confidence: float
    entities: dict
    script: str
    next_action: Literal['play_tts', 'escalate_sim', 'end_call', 'follow_up']
    # test_input is used to simulate STT result when mic is unavailable
    test_input: Optional[str] 

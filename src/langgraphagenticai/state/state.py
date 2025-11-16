from typing_extensions import TypedDict,List
from typing import Annotated
from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional
AllowedIntent = Literal[
    "Billing Issue",
    "SIM Not Working",
    "No Network Coverage",
    "Internet Speed Slow",
    "Data Not Working After Recharge",
    "Call Drops Frequently",
]

class NLUOutput(BaseModel):
    intent: AllowedIntent = Field(description="One of the six canonical intents.")
    confidence: float = Field(ge=0.0, le=1.0, default=0.0, description="0.0..1.0")
    entities: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = None

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

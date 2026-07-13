from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class ParticipantJoinEvent(BaseModel):
    participant_id: str
    display_name: str
    timestamp: float
    email: Optional[str] = Field(default=None)


class ParticipantLeaveEvent(BaseModel):
    participant_id: str
    timestamp: float


class DisplayNameChangeEvent(BaseModel):
    participant_id: str
    new_display_name: str
    timestamp: float


class WebcamStateEvent(BaseModel):
    participant_id: str
    webcam_on: bool
    timestamp: float


class ScreenShareAction(str, Enum):
    STARTED = "started"
    ENDED = "ended"


class ScreenShareEvent(BaseModel):
    participant_id: str
    action: ScreenShareAction
    timestamp: float



class AudioStreamRegistration(BaseModel):
    participant_id: str
    stream_reference: Optional[str] = None
    timestamp: float


class SpeakingAction(str, Enum):
    STARTED = "started"
    ENDED = "ended"


class SpeakingActivityEvent(BaseModel):
    participant_id: str
    action: SpeakingAction
    timestamp: float


class VideoStreamRegistration(BaseModel):
    participant_id: str
    stream_reference: Optional[str] = None
    timestamp: float


class TranscriptEntry(BaseModel):
    participant_id: str
    text: str
    timestamp: float
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0,description="ASR provider confidence (transcription & diarization). NOT the role-classifier confidence.")


class CalendarAttendee(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class CalendarInvite(BaseModel):
    scheduled_start: float = Field(..., description="Meeting-relative or reference timestamp for scheduled start.")
    scheduled_end: Optional[float] = None
    attendees: list[CalendarAttendee] = Field(default_factory=list)


class InterviewSchedule(BaseModel):
    round_number: Optional[int] = None
    expected_duration_minutes: Optional[int] = None
    assigned_interviewers: Optional[list[str]] = None


class ExternalMetadata(BaseModel):
    candidate_name: str
    candidate_email: Optional[str] = None
    interviewer_names: list[str] = Field(default_factory=list)
    calendar_invite: Optional[CalendarInvite] = None
    interview_schedule: Optional[InterviewSchedule] = None


class MeetingInitRequest(BaseModel):
    meeting_id: str
    external_metadata: ExternalMetadata

class ParticipantState(BaseModel):
    participant_id: str
    display_name: str
    display_name_history: list[str] = Field(default_factory=list)
    joined_at: Optional[float] = None
    left_at: Optional[float] = None
    webcam_on: bool = False
    screen_sharing: bool = False
    total_speaking_seconds: float = 0.0
    is_speaking: bool = False
    speaking_started_at: Optional[float] = None 
    transcript_entry_count: int = 0


class MeetingState(BaseModel):
    meeting_id: str
    external_metadata: Optional[ExternalMetadata] = None
    participants: dict[str, ParticipantState] = Field(default_factory=dict)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
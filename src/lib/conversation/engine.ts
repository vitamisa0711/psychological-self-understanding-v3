/**
 * Conversation Engine — PHASE 15 / V2
 *
 * Orchestrates one turn of the multi-turn conversation:
 *   1. Detect signals from the user's message
 *   2. Decide next conversation state
 *   3. Build a prompt appropriate for that state
 *   4. Call the AI provider (via the existing generateControlled path for
 *      generation, or a direct chat completion for conversational response)
 *   5. Update memory
 *   6. Return the response
 *
 * SAFETY: the existing classifySafety() gate still runs on every user
 * message. HIGH/CRITICAL/UNCERTAIN → SAFETY_RESPONSE (no conversation).
 * The Conversation Engine only runs after LOW/MODERATE.
 */

import { detectSignals } from "./signal-detector";
import { nextState, shouldFormulate } from "./state-machine";
import { selectExplorationQuestion } from "./exploration-selector";
import type {
  ConversationEngineInput,
  ConversationEngineOutput,
  ConversationMemory,
  ConversationTurn,
  PartialHypothesis,
} from "./types";

// ─── Memory update ─────────────────────────────────────────────────────────

function updateMemory(
  memory: ConversationMemory,
  params: {
    userMessage: string;
    confirmedObservation?: string;
    deniedHypothesisLabel?: string;
    newHypothesis?: PartialHypothesis;
    questionKey?: string;
    emotions?: string[];
    trigger?: string;
    patternSignals: typeof memory.patternSignals;
    pastExperienceLevel: typeof memory.pastExperienceLevel;
    pastExperienceReported: boolean;
  }
): ConversationMemory {
  const updated: ConversationMemory = {
    ...memory,
    patternSignals: params.patternSignals,
    pastExperienceLevel: params.pastExperienceLevel,
    pastExperienceReported: params.pastExperienceReported,
  };

  if (params.confirmedObservation) {
    updated.confirmedObservations = [
      ...memory.confirmedObservations,
      params.confirmedObservation,
    ];
  }
  if (params.deniedHypothesisLabel) {
    updated.deniedHypotheses = [
      ...memory.deniedHypotheses,
      params.deniedHypothesisLabel,
    ];
    updated.activeHypotheses = memory.activeHypotheses.map((h) =>
      h.label === params.deniedHypothesisLabel ? { ...h, denied: true } : h
    );
  }
  if (params.newHypothesis) {
    const exists = memory.activeHypotheses.some(
      (h) => h.id === params.newHypothesis!.id
    );
    if (!exists) {
      updated.activeHypotheses = [...memory.activeHypotheses, params.newHypothesis];
    }
  }
  if (params.questionKey) {
    updated.questionsAsked = [...memory.questionsAsked, params.questionKey];
  }
  if (params.emotions) {
    const allEmotions = [...new Set([...memory.detectedEmotions, ...params.emotions])];
    updated.detectedEmotions = allEmotions;
  }
  if (params.trigger && !memory.coreTrigger) {
    updated.coreTrigger = params.trigger;
  }
  return updated;
}

// ─── Prompt builder for conversational turns ──────────────────────────────

function buildConversationalPrompt(
  userMessage: string,
  history: ConversationTurn[],
  memory: ConversationMemory,
  _state: "VENTING" | "EXPLORATION" | "REFLECTION" | "FORMULATION" | "NO_FORMULATION_YET",
  _explorationQuestion: { text: string; rationale: string } | null
): { system: string; messages: Array<{ role: string; content: string }> } {
  const deniedNote =
    memory.deniedHypotheses.length > 0
      ? "\n\nNgười dùng đã phủ nhận những cách hiểu sau — không lặp lại: " +
        memory.deniedHypotheses.join(", ")
      : "";

  const system =
    `Bạn đang ngồi cùng một người đang cần được nghe. Không phải bác sĩ, không phải chuyên gia phân tích — một người thật sự lắng nghe và hiểu.

CÁCH BẠN HOẠT ĐỘNG:

Bước 1 — Lắng nghe và an ủi thật sự.
Khi người dùng kể chuyện hoặc bày tỏ cảm xúc, phản chiếu lại đúng những gì họ đang cảm thấy. Thừa nhận cảm xúc đó — không làm nhẹ đi, không phán xét, không vội vào giải thích.
An ủi thật nghĩa là để người đó biết cảm xúc của họ được nhìn thấy. Ví dụ: "Cảm giác bị phớt lờ đúng lúc mình đang cần được nghe — điều đó thực sự rất khó chịu." Không phải "mọi thứ sẽ ổn thôi."
Có thể động viên, cổ vũ khi người dùng đang cố gắng hoặc đang khó khăn.

Bước 2 — Nhận ra phản ứng là bình thường hay bất thường.
Sau khi đã nghe đủ, tự hỏi: phản ứng này có phù hợp với hoàn cảnh không?

BÌNH THƯỜNG — phản ứng khớp với hoàn cảnh:
Ví dụ: bị phản bội → tức giận và đau. Bị phớt lờ → buồn và bực. Mất người thân → đau sâu.
→ Xác nhận điều đó là bình thường. Không cần tìm nguyên nhân sâu xa hơn trừ khi người dùng muốn.

BẤT THƯỜNG — có một trong các dấu hiệu này:
- Cường độ cảm xúc mạnh hơn nhiều so với hoàn cảnh (hoảng loạn vì 30 phút không có tin nhắn)
- Bản thân người dùng thấy phản ứng của mình không hiểu được ("mình biết chuyện nhỏ mà sao...")
- Phản ứng tương tự lặp lại trong nhiều tình huống khác nhau
- Cảm xúc kéo dài, không tự giải quyết được

Bước 3 — Nếu bất thường, gợi mở nhẹ nhàng để tìm ra lý do.
Không hỏi nhiều câu cùng lúc. Một câu, đúng điểm còn thiếu nhất.
Tìm hiểu theo thứ tự từ gần đến xa — không nhảy thẳng vào quá khứ hay trauma:
1. Hoàn cảnh hiện tại: sự việc có thực sự nghiêm trọng hơn người dùng đang nhìn nhận không?
2. Thể trạng gần đây: mệt, stress, thiếu ngủ, chu kỳ kinh nguyệt, áp lực đang chồng chất?
3. Pattern: phản ứng này có hay xảy ra không, với nhiều người hay không?
4. Quá khứ: chỉ hỏi khi có pattern rõ ràng VÀ người dùng tự gợi mở hoặc tự hỏi tại sao.

Bước 4 — Khi đã hiểu đủ, chia sẻ những gì bạn quan sát được.
Không phán xét, không kết luận thay người dùng. Dùng dạng quan sát mở:
"Mình để ý rằng... Điều này có đúng với bạn không?"
Nếu người dùng nói không — cập nhật cách hiểu, không bảo vệ quan sát ban đầu.

NHỮNG GÌ TUYỆT ĐỐI KHÔNG LÀM:
- Không đưa lời khuyên ("bạn nên", "hãy thử", "tốt nhất là", "bạn cần")
- Không định hướng hành vi ("lần sau bạn có thể", "thử nói chuyện với anh ấy")
- Không chẩn đoán ("bạn bị", "bạn có vấn đề", "đây là dấu hiệu của")
- Không tự kết luận người khác có ý gì ("anh ấy cố tình", "anh ấy không quan tâm bạn")
- Không hỏi lại những điều người dùng đã trả lời rõ
- Không hỏi nhiều câu cùng lúc
- Không kết thúc mỗi tin nhắn bằng "Bạn có muốn chia sẻ thêm không?" — câu này máy móc và không tự nhiên

ĐỘ DÀI VÀ GIỌNG ĐIỆU:
Ngắn khi người dùng đang cảm xúc mạnh — đừng lấn át họ bằng chữ.
Dài hơn khi đang cùng nhìn vào một điều gì đó cụ thể.
Giọng điệu: ấm, chân thật, không quá trang trọng, không sáo rỗng.
Nói như một người thật đang ngồi cùng — không phải như một hệ thống đang xử lý.` + deniedNote;

  const messages: Array<{ role: string; content: string }> = [];
  for (const turn of history.slice(-12)) {
    messages.push({ role: turn.role, content: turn.content });
  }
  messages.push({ role: "user", content: userMessage });

  return { system, messages };
}

// ─── Main engine function ──────────────────────────────────────────────────

export async function runConversationTurn(
  input: ConversationEngineInput
): Promise<ConversationEngineOutput> {
  const { userMessage, history, memory, currentState } = input;

  // 1. Detect signals
  const lastAssistantTurn =
    [...history].reverse().find((t) => t.role === "assistant") ?? null;
  const signals = detectSignals(userMessage, history, memory, lastAssistantTurn);

  // 2. Update memory with new signals
  let updatedMemory = updateMemory(memory, {
    userMessage,
    patternSignals: signals.patternSignals,
    pastExperienceLevel: signals.pastExperienceLevel,
    pastExperienceReported: signals.pastExperienceReported,
    confirmedObservation:
      signals.userSignal === "CONFIRMS_REFLECTION" && lastAssistantTurn
        ? lastAssistantTurn.content.slice(0, 120) // brief record
        : undefined,
    deniedHypothesisLabel:
      signals.userSignal === "DENIES_REFLECTION" &&
      memory.activeHypotheses.find((h) => !h.denied)
        ? memory.activeHypotheses.find((h) => !h.denied)!.label
        : undefined,
  });

  // 3. Decide next state
  const resolvedNextState = nextState({
    currentState,
    userSignal: signals.userSignal,
    patternSignals: signals.patternSignals,
    memory: updatedMemory,
    historyLength: history.length,
  });

  // 4. Select exploration question if needed
  let explorationQuestion = null;
  let questionKey: string | null = null;
  if (resolvedNextState === "EXPLORATION") {
    const hasEmotion =
      updatedMemory.detectedEmotions.length > 0 ||
      /buồn|tức|lo|sợ|xấu\s*hổ|ghen|cô\s*đơn|thất\s*vọng|hoảng/i.test(userMessage);
    const hasTrigger = updatedMemory.coreTrigger !== null;
    const hasThought =
      updatedMemory.confirmedObservations.length > 0 ||
      /nghĩ\s*rằng|nghĩ\s*là|cảm\s*thấy\s*như|như\s*là/i.test(userMessage);

    explorationQuestion = selectExplorationQuestion({
      memory: updatedMemory,
      patternSignals: signals.patternSignals,
      pastExperienceLevel: signals.pastExperienceLevel,
      hasEmotion,
      hasTrigger,
      hasThought,
    });

    if (explorationQuestion) {
      // Extract key from the question pool
      questionKey = explorationQuestion.rationale.includes("emotion")
        ? "emotion_felt"
        : explorationQuestion.rationale.includes("trigger")
          ? "trigger_event"
          : explorationQuestion.rationale.includes("thought")
            ? "inner_thought"
            : explorationQuestion.rationale.includes("repetition")
              ? "pattern_repetition"
              : explorationQuestion.rationale.includes("generalization")
                ? "pattern_generalization"
                : explorationQuestion.rationale.includes("unexplained")
                  ? "unexplained_reaction"
                  : explorationQuestion.rationale.includes("level 2") ||
                      explorationQuestion.rationale.includes("prior general")
                    ? "past_childhood"
                    : explorationQuestion.rationale.includes("level 1")
                      ? "past_similar_feeling"
                      : explorationQuestion.rationale.includes("denied")
                        ? "what_feels_closer"
                        : explorationQuestion.rationale.includes("need")
                          ? "unmet_need"
                          : "exploration";
      updatedMemory = updateMemory(updatedMemory, {
        userMessage,
        patternSignals: signals.patternSignals,
        pastExperienceLevel: signals.pastExperienceLevel,
        pastExperienceReported: signals.pastExperienceReported,
        questionKey: questionKey ?? undefined,
      });
    }
  }

  // 5. Build prompt and call AI
  const { system, messages } = buildConversationalPrompt(
    userMessage,
    history,
    updatedMemory,
    resolvedNextState,
    explorationQuestion
  );

  const response = await callConversationalAI(system, messages);

  // 6. Extract hypothesis if this was a REFLECTION turn
  let hypothesisOffered: string | null = null;
  if (resolvedNextState === "REFLECTION") {
    const match = response.match(/Mình\s+đang\s+để\s+ý\s+([^.!?]+)/);
    if (match) {
      hypothesisOffered = match[1].trim();
      const newHyp: PartialHypothesis = {
        id: `hyp_${Date.now()}`,
        label: hypothesisOffered.slice(0, 80),
        evidence: updatedMemory.confirmedObservations.slice(-2),
        confidence: "LOW",
        denied: false,
      };
      updatedMemory = updateMemory(updatedMemory, {
        userMessage,
        patternSignals: signals.patternSignals,
        pastExperienceLevel: signals.pastExperienceLevel,
        pastExperienceReported: signals.pastExperienceReported,
        newHypothesis: newHyp,
      });
    }
  }

  return {
    response,
    nextState: resolvedNextState,
    updatedMemory,
    questionAsked: explorationQuestion?.text ?? null,
    hypothesisOffered,
  };
}

// ─── AI call ───────────────────────────────────────────────────────────────

async function callConversationalAI(
  system: string,
  messages: Array<{ role: string; content: string }>
): Promise<string> {
  if (process.env.AI_PROVIDER === "mock" || process.env.NODE_ENV === "test") {
    return mockConversationalResponse(messages.at(-1)?.content ?? "");
  }

  const apiKey = process.env.AI_API_KEY;
  const model = process.env.AI_MODEL ?? "gpt-4o-mini";
  if (!apiKey) throw new Error("AI_API_KEY missing");

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      max_tokens: 600,
      temperature: 0.7,
      messages: [{ role: "system", content: system }, ...messages],
    }),
    signal: AbortSignal.timeout(30_000),
  });

  if (!res.ok) {
    const err = await res.text().catch(() => "unknown");
    throw new Error(`OpenAI error ${res.status}: ${err.slice(0, 200)}`);
  }

  const json = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  return json.choices?.[0]?.message?.content?.trim() ?? "";
}

function mockConversationalResponse(userMessage: string): string {
  if (/tại\s*sao|không\s*hiểu/i.test(userMessage)) {
    return "Nghe như bạn đang muốn hiểu rõ hơn điều gì đang xảy ra với mình. Cảm giác đó khiến bạn khó chịu như thế nào?";
  }
  if (/buồn|tức|lo|sợ/i.test(userMessage)) {
    return "Nghe như bạn đang mang một cảm xúc khá nặng. Nếu bạn muốn, bạn có thể kể thêm chuyện gì đã xảy ra không?";
  }
  return "Mình đang nghe. Bạn có muốn kể thêm không?";
}

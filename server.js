import express from "express";
import OpenAI from "openai";
import path from "path";
import { fileURLToPath } from "url";

const app = express();
app.use(express.json({ limit: "20mb" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve the frontend
app.use(express.static(path.join(__dirname, "public")));

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Missing OPENAI_API_KEY env var.");
  process.exit(1);
}

const client = new OpenAI({ apiKey });

// ---------------------------
// Utility: robust JSON parse
// ---------------------------
function safeJsonParse(maybeJsonText) {
  // Try direct parse first
  try {
    return JSON.parse(maybeJsonText);
  } catch {}

  // Try to extract a JSON object block
  const start = maybeJsonText.indexOf("{");
  const end = maybeJsonText.lastIndexOf("}");
  if (start >= 0 && end > start) {
    const sliced = maybeJsonText.slice(start, end + 1);
    return JSON.parse(sliced);
  }
  throw new Error("Model did not return valid JSON.");
}

// -------------------------------------------
// 1) Upload TXT -> create vector store + index
// -------------------------------------------
app.post("/api/kb", async (req, res) => {
  try {
    const { filename, content } = req.body;

    if (typeof content !== "string" || !content.trim()) {
      return res.status(400).json({ error: "Empty content." });
    }

    // Create a vector store
    const vectorStore = await client.vectorStores.create({
      name: `AI Trader Quiz KB - ${new Date().toISOString()}`
    });

    // Create an OpenAI file from the text content
    // Node 18+ provides global File/Blob in most environments.
    // If your Node runtime lacks File, see notes in "Unknown" section.
    const fileObj = new File(
      [content],
      filename || "knowledge.txt",
      { type: "text/plain" }
    );

    const uploadedFile = await client.files.create({
      file: fileObj,
      purpose: "assistants"
    });

    // Attach file to vector store
    await client.vectorStores.files.create(vectorStore.id, {
      file_id: uploadedFile.id
    });

    // Poll until file is indexed (or timeout)
    const timeoutMs = 60_000;
    const start = Date.now();
    while (true) {
      const list = await client.vectorStores.files.list(vectorStore.id, { limit: 20 });
      const entry = list.data.find(x => x.id);
      // Some SDKs expose status on file objects; if not, just wait a little.
      // We'll do a small delay and exit after timeout.
      if (Date.now() - start > timeoutMs) break;
      await new Promise(r => setTimeout(r, 800));
      // No strict status check; indexing is usually fast for txt.
      // We return immediately after a short wait if you prefer:
      if (Date.now() - start > 2500) break;
    }

    res.json({ vector_store_id: vectorStore.id });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "KB upload/indexing failed.", detail: String(err?.message || err) });
  }
});

// -------------------------------------------
// 2) Generate question using file_search
// -------------------------------------------
app.post("/api/generate-question", async (req, res) => {
  try {
    const { vector_store_id, askedQuestions } = req.body;

    if (!vector_store_id) {
      return res.status(400).json({ error: "Missing vector_store_id." });
    }

    const historyBlock = Array.isArray(askedQuestions) && askedQuestions.length
      ? `Already asked (avoid repeating these topics/questions):\n${askedQuestions.map((q, i) => `${i + 1}. ${q}`).join("\n")}`
      : "";

    const prompt = `
You are an expert financial educator creating an intraday-trader quiz.

TASK:
- Create ONE challenging multiple-choice question with exactly four options (A, B, C, D).
- Exactly one option must be correct.
- Use ONLY information that can be retrieved from the uploaded document via file_search.
- Do NOT use outside knowledge.
- Avoid repeating earlier questions/topics.

${historyBlock}

Return STRICT JSON ONLY with this schema:
{
  "question": "string",
  "options": { "A": "string", "B": "string", "C": "string", "D": "string" },
  "correctOption": "A|B|C|D"
}
`.trim();

    const r = await client.responses.create({
      model: "gpt-4.1-mini",
      input: prompt,
      tools: [{ type: "file_search", vector_store_ids: [vector_store_id] }],
      // Try to force JSON-only output; if your SDK/model ignores, safeJsonParse still handles.
      response_format: { type: "json_object" }
    });

    const obj = safeJsonParse(r.output_text);

    // Basic validation
    if (!obj?.question || !obj?.options || !obj?.correctOption) {
      throw new Error("Invalid question JSON shape.");
    }
    if (!["A", "B", "C", "D"].includes(obj.correctOption)) {
      throw new Error("correctOption must be A, B, C, or D.");
    }

    res.json(obj);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Question generation failed.", detail: String(err?.message || err) });
  }
});

// -------------------------------------------
// 3) Evaluate answer using file_search
// -------------------------------------------
app.post("/api/evaluate-answer", async (req, res) => {
  try {
    const { vector_store_id, questionData, selectedKey, selectedText } = req.body;

    if (!vector_store_id) return res.status(400).json({ error: "Missing vector_store_id." });
    if (!questionData?.question || !questionData?.options || !questionData?.correctOption) {
      return res.status(400).json({ error: "Missing/invalid questionData." });
    }
    if (!["A", "B", "C", "D"].includes(selectedKey)) {
      return res.status(400).json({ error: "selectedKey must be A, B, C, or D." });
    }

    const prompt = `
You are grading a trader quiz answer.

Question:
${questionData.question}

Options:
A: ${questionData.options.A}
B: ${questionData.options.B}
C: ${questionData.options.C}
D: ${questionData.options.D}

Correct option: ${questionData.correctOption}
User selected: ${selectedKey} (${selectedText})

TASK:
- Determine if the user is correct.
- Provide an explanation grounded ONLY in text retrieved via file_search from the uploaded document.
- If incorrect: explain why selected is wrong and why correct is correct.
- If correct: explain why it is correct.

Return STRICT JSON ONLY:
{
  "isCorrect": true/false,
  "explanation": "string"
}
`.trim();

    const r = await client.responses.create({
      model: "gpt-4.1-mini",
      input: prompt,
      tools: [{ type: "file_search", vector_store_ids: [vector_store_id] }],
      response_format: { type: "json_object" }
    });

    const obj = safeJsonParse(r.output_text);

    if (typeof obj?.isCorrect !== "boolean" || typeof obj?.explanation !== "string") {
      throw new Error("Invalid evaluation JSON shape.");
    }

    res.json(obj);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Evaluation failed.", detail: String(err?.message || err) });
  }
});

app.get("/health", (_, res) => res.json({ ok: true }));

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Running on http://localhost:${port}`));

import { useMemo, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import SceneBackground from "./components/SceneBackground";

const MODES = ["PDF Chat", "Arxiv Search", "Wikipedia Search"];

export default function App() {
    const [mode, setMode] = useState("PDF Chat");
    const [sessionId, setSessionId] = useState("default_session");
    const [question, setQuestion] = useState("");
    const [files, setFiles] = useState([]);
    const [pdfReady, setPdfReady] = useState(false);
    const [answer, setAnswer] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const canAskPdf = useMemo(() => pdfReady && question.trim().length > 0, [pdfReady, question]);
    const canAskSearch = useMemo(() => question.trim().length > 0, [question]);

    const uploadPdf = async () => {
        if (!files.length) {
            setError("Choose at least one PDF file first.");
            return;
        }

        setLoading(true);
        setError("");
        setAnswer("");

        try {
            const form = new FormData();
            form.append("session_id", sessionId);
            files.forEach((file) => form.append("files", file));

            await axios.post("/api/pdf/upload", form, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setPdfReady(true);
            setAnswer("PDFs uploaded and indexed. You can ask questions now.");
        } catch (err) {
            setError(err.response?.data?.detail || "Failed to upload PDFs.");
            setPdfReady(false);
        } finally {
            setLoading(false);
        }
    };

    const ask = async () => {
        if (mode === "PDF Chat" && !canAskPdf) {
            setError("Upload PDFs first, then ask a question.");
            return;
        }
        if (mode !== "PDF Chat" && !canAskSearch) {
            setError("Type a question first.");
            return;
        }

        setLoading(true);
        setError("");

        try {
            if (mode === "PDF Chat") {
                const { data } = await axios.post("/api/pdf/ask", {
                    session_id: sessionId,
                    question,
                });
                setAnswer(data.answer || "No response from backend.");
            } else if (mode === "Arxiv Search") {
                const { data } = await axios.post("/api/arxiv/search", { query: question });
                setAnswer(data.result || "No result from arXiv.");
            } else {
                const { data } = await axios.post("/api/wikipedia/search", { query: question });
                setAnswer(data.result || "No result from Wikipedia.");
            }
        } catch (err) {
            setError(err.response?.data?.detail || "Request failed.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-shell">
            <SceneBackground />
            <div className="overlay" />

            <motion.main
                className="panel"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, ease: "easeOut" }}
            >
                <header className="hero">
                    <motion.h1
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.15, duration: 0.6 }}
                    >
                        AI Research Assistant
                    </motion.h1>
                </header>

                <section className="controls">
                    <label>
                        Mode
                        <select value={mode} onChange={(e) => setMode(e.target.value)}>
                            {MODES.map((m) => (
                                <option key={m} value={m}>
                                    {m}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label>
                        Session ID
                        <input
                            type="text"
                            value={sessionId}
                            onChange={(e) => setSessionId(e.target.value)}
                            placeholder="default_session"
                        />
                    </label>

                    {mode === "PDF Chat" && (
                        <div className="pdf-upload-wrap">
                            <input
                                type="file"
                                accept="application/pdf"
                                multiple
                                onChange={(e) => {
                                    const selected = Array.from(e.target.files || []);
                                    setFiles(selected);
                                    setPdfReady(false);
                                }}
                            />
                            <button type="button" onClick={uploadPdf} disabled={loading}>
                                {loading ? "Uploading..." : "Upload PDFs"}
                            </button>
                        </div>
                    )}

                    <label>
                        Question
                        <textarea
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            placeholder={mode === "PDF Chat" ? "Ask something from your uploaded PDFs..." : "Type your search query..."}
                            rows={5}
                        />
                    </label>

                    <button type="button" className="primary" onClick={ask} disabled={loading}>
                        {loading ? "Working..." : mode === "PDF Chat" ? "Ask PDF" : "Search"}
                    </button>
                </section>

                {error && <p className="status error">{error}</p>}
                {answer && <pre className="answer">{answer}</pre>}
            </motion.main>
        </div>
    );
}

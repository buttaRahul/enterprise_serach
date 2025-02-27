import React, { useState } from 'react';
import Paper from '@mui/material/Paper';
import { Typography, Box, TextField, Button, Container } from '@mui/material';

const ChatBot = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSend = async () => {
        if (input.trim() === "") return;

        const newMessages = [...messages, { sender: "user", text: input }];
        setMessages(newMessages);
        setInput("");
        setLoading(true);

        try {
            const response = await fetch("http://127.0.0.1:8000/submit-query/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query: input })
            });

            if (!response.ok) {
                throw new Error("Failed to fetch response");
            }

            const data = await response.json();
            setMessages((prev) => [...prev, { sender: "bot", text: data.llm_response }]);
        } catch (error) {
            console.error("Error fetching response:", error);
            setMessages((prev) => [...prev, { sender: "bot", text: "Error getting response. Please try again." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: '#e0e8fc' }}>
            <Paper elevation={6} sx={{ p: 3, width: '70%', height: 550, display: "flex", flexDirection: "column", borderRadius: 3, backgroundColor: '#ffffff' }}>
                <Typography variant="h5" align="center" gutterBottom sx={{ color: '#2c3e50', fontWeight: 'bold' }}>
                    Enterprise Search Bot
                </Typography>
                <Box sx={{ flexGrow: 1, overflowY: "auto", mb: 2, p: 2, border: "1px solid #ddd", borderRadius: 2, backgroundColor: '#f9f9fb', maxHeight: 400 }}>
                    {messages.map((msg, index) => (
                        <Typography
                            key={index}
                            sx={{
                                textAlign: msg.sender === "user" ? "right" : "left",
                                backgroundColor: msg.sender === "user" ? "#d1ecf1" : "#f1f1f1",
                                p: 1.5,
                                borderRadius: 2,
                                my: 0.5,
                                display: "inline-block",
                                maxWidth: '80%',
                                whiteSpace: "pre-wrap" // Preserves line breaks and spacing
                            }}
                        >
                            {msg.text}
                        </Typography>
                    ))}
                </Box>
                <Box sx={{ display: "flex", gap: 1 }}>
                    <TextField
                        fullWidth
                        variant="outlined"
                        size="small"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type a message..."
                        disabled={loading}
                    />
                    <Button variant="contained" onClick={handleSend} disabled={loading} sx={{ backgroundColor: '#2c3e50', color: '#fff' }}>
                        {loading ? "Sending..." : "Send"}
                    </Button>
                </Box>
            </Paper>
        </Container>
    );
};

export default ChatBot;

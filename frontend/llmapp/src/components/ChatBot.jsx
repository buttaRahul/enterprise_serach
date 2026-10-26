import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Box, Typography, TextField, IconButton, Chip, Collapse,
  CircularProgress, Tooltip, Avatar, Fade,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ThumbUpOutlinedIcon from '@mui/icons-material/ThumbUpOutlined';
import ThumbDownOutlinedIcon from '@mui/icons-material/ThumbDownOutlined';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import EmailIcon from '@mui/icons-material/Email';
import StorageIcon from '@mui/icons-material/Storage';
import LanguageIcon from '@mui/icons-material/Language';
import BusinessIcon from '@mui/icons-material/Business';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import SearchIcon from '@mui/icons-material/Search';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

const API = 'http://127.0.0.1:8000';

const SIDEBAR_W = 260;
const BRAND_COLOR = '#0f172a';
const ACCENT = '#6366f1';

const SOURCE_META = {
  email:  { icon: <EmailIcon sx={{ fontSize: 13 }} />,   color: '#3b82f6', bg: '#eff6ff', label: 'Email'    },
  sql:    { icon: <StorageIcon sx={{ fontSize: 13 }} />,  color: '#10b981', bg: '#f0fdf4', label: 'Database' },
  web:    { icon: <LanguageIcon sx={{ fontSize: 13 }} />, color: '#f59e0b', bg: '#fffbeb', label: 'Web'      },
  site:   { icon: <BusinessIcon sx={{ fontSize: 13 }} />, color: '#8b5cf6', bg: '#f5f3ff', label: 'Site'     },
};

const SUGGESTIONS = [
  { text: 'Find emails about pull requests', icon: <EmailIcon sx={{ fontSize: 14 }} /> },
  { text: 'How many employees are in Engineering?', icon: <StorageIcon sx={{ fontSize: 14 }} /> },
  { text: 'List all projects with their budget', icon: <StorageIcon sx={{ fontSize: 14 }} /> },
  { text: 'Find emails about deployment failures', icon: <EmailIcon sx={{ fontSize: 14 }} /> },
  { text: 'What does the company do?', icon: <BusinessIcon sx={{ fontSize: 14 }} /> },
  { text: 'Show me recent emails', icon: <EmailIcon sx={{ fontSize: 14 }} /> },
];

// ── Source badge ──────────────────────────────────────────────────────────────

function SourceBadge({ source }) {
  const m = SOURCE_META[source] || { icon: <SearchIcon sx={{ fontSize: 13 }} />, color: '#6b7280', bg: '#f3f4f6', label: source };
  return (
    <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.4, px: 0.8, py: 0.2,
      bgcolor: m.bg, color: m.color, borderRadius: 1, fontSize: 11, fontWeight: 600, border: `1px solid ${m.color}22` }}>
      {m.icon}{m.label}
    </Box>
  );
}

// ── Evidence card ─────────────────────────────────────────────────────────────

function EvidenceCard({ item }) {
  const [open, setOpen] = useState(false);
  const source = item.source || 'unknown';
  const rawId = (item.id || '').replace(/^email:/, '');
  const m = SOURCE_META[source];

  const title = source === 'email'
    ? (item.sender || rawId)
    : source === 'sql' ? 'Query result'
    : source === 'site' ? 'Company site'
    : 'Web result';

  const preview = item.content
    ? item.content.slice(0, 100) + (item.content.length > 100 ? '…' : '')
    : item.rows ? `${item.rows.length} row${item.rows.length !== 1 ? 's' : ''} returned`
    : '';

  return (
    <Box sx={{ mb: 0.5, border: '1px solid', borderColor: m ? `${m.color}33` : '#e5e7eb',
      borderRadius: 1.5, overflow: 'hidden', bgcolor: '#fff' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', px: 1.5, py: 0.75, cursor: 'pointer',
        '&:hover': { bgcolor: '#f8fafc' }, gap: 1 }}
        onClick={() => setOpen(o => !o)}>
        <SourceBadge source={source} />
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography sx={{ fontSize: 12, fontWeight: 600, color: '#374151', lineHeight: 1.3 }}>{title}</Typography>
          {preview && !open && (
            <Typography sx={{ fontSize: 11, color: '#6b7280', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {preview}
            </Typography>
          )}
        </Box>
        {open ? <ExpandLessIcon sx={{ fontSize: 15, color: '#9ca3af', flexShrink: 0 }} />
               : <ExpandMoreIcon sx={{ fontSize: 15, color: '#9ca3af', flexShrink: 0 }} />}
      </Box>
      <Collapse in={open}>
        <Box sx={{ px: 1.5, pb: 1.5, borderTop: '1px solid #f3f4f6' }}>
          {/* SQL query badge */}
          {source === 'sql' && item.title && (
            <Box sx={{ mt: 1 }}>
              <Typography sx={{ fontSize: 10, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: 0.5, mb: 0.5 }}>
                Query executed
              </Typography>
              <Box component="pre" sx={{ fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                m: 0, p: 1, bgcolor: '#0f172a', color: '#a5f3fc', borderRadius: 1.5, fontFamily: 'monospace',
                border: '1px solid #1e293b', lineHeight: 1.6 }}>
                {item.title}
              </Box>
            </Box>
          )}
          {item.rows && (
            <Box component="pre" sx={{ fontSize: 11, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              m: 0, mt: 1, p: 1, bgcolor: '#f8fafc', borderRadius: 1, fontFamily: 'monospace', color: '#374151' }}>
              {JSON.stringify(item.rows.slice(0, 5), null, 2)}
              {item.rows.length > 5 && `\n… ${item.rows.length - 5} more rows`}
            </Box>
          )}
          {/* SQL result as markdown table (stored in item.text) */}
          {source === 'sql' && item.text && item.text !== '_(no rows)_' && (
            <Box sx={{ mt: 1, fontSize: 12, color: '#374151',
              '& table': { borderCollapse: 'collapse', width: '100%', fontSize: 12 },
              '& th': { bgcolor: '#f3f4f6', px: 1, py: 0.5, textAlign: 'left', fontWeight: 600, border: '1px solid #e5e7eb' },
              '& td': { px: 1, py: 0.5, border: '1px solid #e5e7eb' },
            }}>
              <ReactMarkdown>{item.text}</ReactMarkdown>
            </Box>
          )}
          {item.content && (
            <Typography sx={{ fontSize: 12, mt: 1, color: '#4b5563', lineHeight: 1.6 }}>{item.content}</Typography>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}

// ── Feedback ──────────────────────────────────────────────────────────────────

function FeedbackRow({ threadId, question, evidence }) {
  const [rating, setRating] = useState(null);
  const [showCorrection, setShowCorrection] = useState(false);
  const inputRef = useRef();
  const sqlEvidence = (evidence || []).find(e => e.source === 'sql');

  const submit = async (r) => {
    if (rating) return;
    setRating(r);
    if (r === 'down') { setShowCorrection(true); return; }
    await fetch(`${API}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId, question, sql: sqlEvidence?.sql || '', rating: r }),
    }).catch(() => {});
  };

  const submitCorrection = async () => {
    const correction = inputRef.current?.value || '';
    setShowCorrection(false);
    await fetch(`${API}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId, question, sql: sqlEvidence?.sql || '', rating: 'down', correction }),
    }).catch(() => {});
  };

  return (
    <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
      <Typography sx={{ fontSize: 11, color: '#9ca3af', mr: 0.5 }}>Was this helpful?</Typography>
      <Tooltip title="Yes, helpful">
        <IconButton size="small" onClick={() => submit('up')}
          sx={{ p: 0.5, color: rating === 'up' ? '#10b981' : '#d1d5db', '&:hover': { color: '#10b981' } }}>
          {rating === 'up' ? <ThumbUpIcon sx={{ fontSize: 15 }} /> : <ThumbUpOutlinedIcon sx={{ fontSize: 15 }} />}
        </IconButton>
      </Tooltip>
      <Tooltip title="Not helpful">
        <IconButton size="small" onClick={() => submit('down')}
          sx={{ p: 0.5, color: rating === 'down' ? '#ef4444' : '#d1d5db', '&:hover': { color: '#ef4444' } }}>
          {rating === 'down' ? <ThumbDownIcon sx={{ fontSize: 15 }} /> : <ThumbDownOutlinedIcon sx={{ fontSize: 15 }} />}
        </IconButton>
      </Tooltip>
      {showCorrection && (
        <Fade in>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, ml: 0.5 }}>
            <TextField size="small" placeholder="What was wrong? (optional)" inputRef={inputRef}
              sx={{ '& .MuiInputBase-input': { fontSize: 12, py: 0.5 } }}
              onKeyDown={e => e.key === 'Enter' && submitCorrection()} />
            <Chip label="Send" size="small" onClick={submitCorrection}
              sx={{ cursor: 'pointer', bgcolor: ACCENT, color: '#fff', fontSize: 11 }} />
          </Box>
        </Fade>
      )}
    </Box>
  );
}

// ── Trace pills ───────────────────────────────────────────────────────────────

function TracePills({ trace }) {
  const [open, setOpen] = useState(false);
  if (!trace || trace.length === 0) return null;
  const total = trace.reduce((s, t) => s + (t.ms || 0), 0);
  return (
    <Box sx={{ mt: 0.75 }}>
      <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5, cursor: 'pointer',
        color: '#9ca3af', '&:hover': { color: '#6b7280' } }} onClick={() => setOpen(o => !o)}>
        <AccessTimeIcon sx={{ fontSize: 13 }} />
        <Typography sx={{ fontSize: 11 }}>{total}ms · {trace.length} nodes</Typography>
        {open ? <ExpandLessIcon sx={{ fontSize: 13 }} /> : <ExpandMoreIcon sx={{ fontSize: 13 }} />}
      </Box>
      <Collapse in={open}>
        <Box sx={{ mt: 0.5, display: 'flex', flexWrap: 'wrap', gap: 0.4 }}>
          {trace.map((t, i) => (
            <Box key={i} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.4,
              px: 0.8, py: 0.3, bgcolor: '#f3f4f6', borderRadius: 1, fontSize: 11, color: '#4b5563' }}>
              <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: ACCENT, flexShrink: 0 }} />
              {t.node} {t.ms != null && <span style={{ color: '#9ca3af' }}>{t.ms}ms</span>}
            </Box>
          ))}
        </Box>
      </Collapse>
    </Box>
  );
}

// ── Typing indicator ──────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, px: 0.5 }}>
      {[0, 1, 2].map(i => (
        <Box key={i} sx={{
          width: 7, height: 7, borderRadius: '50%', bgcolor: '#9ca3af',
          animation: 'bounce 1.2s infinite ease-in-out',
          animationDelay: `${i * 0.2}s`,
          '@keyframes bounce': {
            '0%, 80%, 100%': { transform: 'scale(0.8)', opacity: 0.5 },
            '40%': { transform: 'scale(1.2)', opacity: 1 },
          },
        }} />
      ))}
    </Box>
  );
}

// ── Welcome screen ────────────────────────────────────────────────────────────

function WelcomeScreen({ onSuggest }) {
  const cards = [
    { icon: <EmailIcon sx={{ fontSize: 28, color: '#3b82f6' }} />, title: 'Email search', desc: 'Find emails by topic, sender, or date range' },
    { icon: <StorageIcon sx={{ fontSize: 28, color: '#10b981' }} />, title: 'Database queries', desc: 'Ask about employees, projects, and tickets in plain English' },
    { icon: <BusinessIcon sx={{ fontSize: 28, color: '#8b5cf6' }} />, title: 'Company knowledge', desc: 'Search the company website and documentation' },
  ];
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', pt: 4, pb: 2, px: 2 }}>
      <Box sx={{ mb: 3, textAlign: 'center' }}>
        <Box sx={{ width: 52, height: 52, borderRadius: 3, bgcolor: ACCENT, display: 'flex',
          alignItems: 'center', justifyContent: 'center', mx: 'auto', mb: 1.5, boxShadow: `0 4px 14px ${ACCENT}44` }}>
          <AutoAwesomeIcon sx={{ color: '#fff', fontSize: 28 }} />
        </Box>
        <Typography sx={{ fontWeight: 700, fontSize: 20, color: '#111827' }}>How can I help you?</Typography>
        <Typography sx={{ fontSize: 13, color: '#6b7280', mt: 0.5 }}>
          Search across emails, databases, and company knowledge
        </Typography>
      </Box>

      <Box sx={{ display: 'flex', gap: 1.5, mb: 3, flexWrap: 'wrap', justifyContent: 'center' }}>
        {cards.map((c, i) => (
          <Box key={i} sx={{ width: 160, p: 1.5, bgcolor: '#fff', borderRadius: 2,
            border: '1px solid #e5e7eb', textAlign: 'center', boxShadow: '0 1px 4px #0001' }}>
            {c.icon}
            <Typography sx={{ fontSize: 12, fontWeight: 600, color: '#374151', mt: 0.5 }}>{c.title}</Typography>
            <Typography sx={{ fontSize: 11, color: '#6b7280', mt: 0.25 }}>{c.desc}</Typography>
          </Box>
        ))}
      </Box>

      <Typography sx={{ fontSize: 12, color: '#9ca3af', mb: 1 }}>Try asking…</Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, justifyContent: 'center', maxWidth: 540 }}>
        {SUGGESTIONS.map((s, i) => (
          <Chip key={i} icon={s.icon} label={s.text} size="small" variant="outlined" onClick={() => onSuggest(s.text)}
            sx={{ fontSize: 11, cursor: 'pointer', borderColor: '#e5e7eb', color: '#374151',
              '&:hover': { bgcolor: '#f3f4f6', borderColor: ACCENT, color: ACCENT } }} />
        ))}
      </Box>
    </Box>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ msg, threadId }) {
  const isUser = msg.sender === 'user';
  const isClarify = msg.status === 'needs_clarification';

  return (
    <Fade in>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5, mb: 2,
        flexDirection: isUser ? 'row-reverse' : 'row' }}>
        {/* Avatar */}
        <Avatar sx={{ width: 32, height: 32, flexShrink: 0,
          bgcolor: isUser ? '#e0e7ff' : ACCENT,
          color: isUser ? ACCENT : '#fff', fontSize: 14, fontWeight: 700 }}>
          {isUser ? '?' : <AutoAwesomeIcon sx={{ fontSize: 16 }} />}
        </Avatar>

        <Box sx={{ maxWidth: '78%', minWidth: 0 }}>
          {/* Bubble */}
          <Box sx={{
            px: 2, py: 1.5, borderRadius: isUser ? '18px 4px 18px 18px' : '4px 18px 18px 18px',
            bgcolor: isUser ? ACCENT : '#fff',
            color: isUser ? '#fff' : '#111827',
            boxShadow: isUser ? `0 2px 8px ${ACCENT}44` : '0 1px 4px #0001',
            border: isClarify ? '2px solid #f59e0b' : isUser ? 'none' : '1px solid #f3f4f6',
          }}>
            {isUser ? (
              <Typography sx={{ fontSize: 14, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{msg.text}</Typography>
            ) : msg.streaming && !msg.text ? (
              <TypingDots />
            ) : (
              <Box sx={{
                fontSize: 14, lineHeight: 1.7, color: 'inherit',
                '& p': { m: 0, mb: 0.75 }, '& p:last-child': { mb: 0 },
                '& ul, & ol': { m: 0, pl: 2.5 }, '& li': { mb: 0.5 },
                '& strong': { fontWeight: 600 },
                '& code': { bgcolor: '#f3f4f6', px: 0.6, borderRadius: 0.5, fontSize: 12.5, fontFamily: 'monospace' },
              }}>
                <ReactMarkdown>{msg.text}</ReactMarkdown>
                {msg.streaming && <CircularProgress size={12} sx={{ ml: 0.5, verticalAlign: 'middle', color: '#9ca3af' }} />}
              </Box>
            )}
          </Box>

          {/* Evidence */}
          {!isUser && !msg.streaming && msg.evidence && msg.evidence.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography sx={{ fontSize: 11, color: '#9ca3af', mb: 0.5, fontWeight: 500 }}>
                {msg.evidence.length} source{msg.evidence.length !== 1 ? 's' : ''} used
              </Typography>
              {msg.evidence.map((ev, i) => <EvidenceCard key={i} item={ev} />)}
            </Box>
          )}

          {/* Trace + feedback */}
          {!isUser && !msg.streaming && (
            <>
              <TracePills trace={msg.trace} />
              {!isClarify && <FeedbackRow threadId={threadId} question={msg.question} evidence={msg.evidence} />}
            </>
          )}
        </Box>
      </Box>
    </Fade>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({ onSuggest, messageCount }) {
  return (
    <Box sx={{ width: SIDEBAR_W, flexShrink: 0, bgcolor: BRAND_COLOR, display: 'flex',
      flexDirection: 'column', height: '100vh', p: 2.5 }}>
      {/* Brand */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
        <Box sx={{ width: 36, height: 36, borderRadius: 2, bgcolor: ACCENT, display: 'flex',
          alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <SearchIcon sx={{ color: '#fff', fontSize: 20 }} />
        </Box>
        <Box>
          <Typography sx={{ color: '#fff', fontWeight: 700, fontSize: 15, lineHeight: 1.2 }}>Enterprise</Typography>
          <Typography sx={{ color: '#6366f1', fontWeight: 600, fontSize: 13, lineHeight: 1.2 }}>Search</Typography>
        </Box>
      </Box>

      {/* Sources */}
      <Typography sx={{ color: '#475569', fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: 0.8, mb: 1.5 }}>Data sources</Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75, mb: 3 }}>
        {Object.entries(SOURCE_META).map(([key, m]) => (
          <Box key={key} sx={{ display: 'flex', alignItems: 'center', gap: 1.25, px: 1, py: 0.6,
            borderRadius: 1.5, bgcolor: '#ffffff08' }}>
            <Box sx={{ color: m.color, display: 'flex' }}>{m.icon}</Box>
            <Typography sx={{ color: '#cbd5e1', fontSize: 12 }}>{m.label}</Typography>
          </Box>
        ))}
      </Box>

      {/* Suggested queries */}
      <Typography sx={{ color: '#475569', fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: 0.8, mb: 1.5 }}>Try asking</Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, flexGrow: 1 }}>
        {SUGGESTIONS.map((s, i) => (
          <Box key={i} onClick={() => onSuggest(s.text)} sx={{
            px: 1.25, py: 0.75, borderRadius: 1.5, cursor: 'pointer', display: 'flex',
            alignItems: 'flex-start', gap: 1,
            '&:hover': { bgcolor: '#ffffff12' }, transition: 'background 0.15s',
          }}>
            <Box sx={{ color: '#475569', mt: 0.1, flexShrink: 0 }}>{s.icon}</Box>
            <Typography sx={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.4 }}>{s.text}</Typography>
          </Box>
        ))}
      </Box>

      {/* Session info */}
      <Box sx={{ mt: 'auto', pt: 2, borderTop: '1px solid #ffffff12' }}>
        <Typography sx={{ color: '#475569', fontSize: 11 }}>
          {messageCount} message{messageCount !== 1 ? 's' : ''} this session
        </Typography>
      </Box>
    </Box>
  );
}

// ── Main ChatBot ──────────────────────────────────────────────────────────────

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [threadId] = useState(() => `thread-${Math.random().toString(36).slice(2)}`);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const appendBot = (patch) =>
    setMessages(prev => {
      const copy = [...prev];
      const last = copy[copy.length - 1];
      if (last && last.sender === 'bot' && last.streaming) {
        copy[copy.length - 1] = { ...last, ...patch };
      } else {
        copy.push({ sender: 'bot', text: '', streaming: true, ...patch });
      }
      return copy;
    });

  const handleSend = async (queryOverride) => {
    const query = (queryOverride || input).trim();
    if (!query || loading) return;
    setInput('');
    setLoading(true);
    inputRef.current?.focus();

    setMessages(prev => [...prev, { sender: 'user', text: query }]);
    appendBot({ text: '', streaming: true });

    try {
      const res = await fetch(`${API}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, thread_id: threadId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalAnswer = '';
      let finalEvidence = [];
      let finalTrace = [];
      let finalStatus = 'answered';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === 'answer') { finalAnswer = event.text; appendBot({ text: finalAnswer, streaming: true }); }
            else if (event.type === 'evidence') { finalEvidence = [...finalEvidence, event.item]; }
            else if (event.type === 'node_end')  { finalTrace = [...finalTrace, event]; }
            else if (event.type === 'done')       { finalStatus = event.status || 'answered'; }
          } catch { /* skip */ }
        }
      }
      appendBot({ text: finalAnswer || '(no answer)', streaming: false, evidence: finalEvidence, trace: finalTrace, status: finalStatus, question: query });
    } catch {
      try {
        const res = await fetch(`${API}/api/chat`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, thread_id: threadId }),
        });
        const data = await res.json();
        appendBot({ text: data.answer || 'No answer.', streaming: false, evidence: data.evidence || [], trace: data.trace || [], status: data.status || 'answered', question: query });
      } catch {
        appendBot({ text: 'Could not reach the server. Is the backend running?', streaming: false });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh', bgcolor: '#f8fafc', overflow: 'hidden' }}>
      <Sidebar onSuggest={q => handleSend(q)} messageCount={messages.filter(m => m.sender === 'user').length} />

      {/* Main panel */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Top bar */}
        <Box sx={{ px: 3, py: 1.75, bgcolor: '#fff', borderBottom: '1px solid #e5e7eb',
          display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#10b981',
            boxShadow: '0 0 0 3px #10b98122' }} />
          <Typography sx={{ fontWeight: 600, color: '#374151', fontSize: 14 }}>New conversation</Typography>
          <Chip label={`ID: ${threadId.slice(7, 15)}`} size="small"
            sx={{ fontSize: 10, color: '#9ca3af', bgcolor: '#f3f4f6', ml: 'auto' }} />
        </Box>

        {/* Messages */}
        <Box sx={{ flexGrow: 1, overflowY: 'auto', px: 3, py: 2 }}>
          {messages.length === 0
            ? <WelcomeScreen onSuggest={q => handleSend(q)} />
            : messages.map((msg, i) => <MessageBubble key={i} msg={msg} threadId={threadId} />)
          }
          <div ref={bottomRef} />
        </Box>

        {/* Input bar */}
        <Box sx={{ px: 3, py: 2, bgcolor: '#fff', borderTop: '1px solid #e5e7eb' }}>
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'flex-end',
            bgcolor: '#f8fafc', border: '1.5px solid', borderColor: loading ? ACCENT : '#e5e7eb',
            borderRadius: 3, px: 2, py: 1, transition: 'border-color 0.2s',
            '&:focus-within': { borderColor: ACCENT, boxShadow: `0 0 0 3px ${ACCENT}22` } }}>
            <TextField
              inputRef={inputRef}
              fullWidth multiline maxRows={4} variant="standard"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
              placeholder="Ask anything — emails, data, company info…"
              disabled={loading}
              InputProps={{ disableUnderline: true, style: { fontSize: 14, lineHeight: 1.6 } }}
            />
            <IconButton onClick={() => handleSend()} disabled={loading || !input.trim()} size="small"
              sx={{ bgcolor: loading || !input.trim() ? '#e5e7eb' : ACCENT,
                color: loading || !input.trim() ? '#9ca3af' : '#fff',
                '&:hover': { bgcolor: loading || !input.trim() ? '#e5e7eb' : '#4f46e5' },
                transition: 'all 0.2s', flexShrink: 0, width: 36, height: 36 }}>
              {loading ? <CircularProgress size={16} sx={{ color: ACCENT }} /> : <SendIcon sx={{ fontSize: 18 }} />}
            </IconButton>
          </Box>
          <Typography sx={{ fontSize: 11, color: '#9ca3af', mt: 0.75, textAlign: 'center' }}>
            Enter to send · Shift+Enter for new line
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatBot;

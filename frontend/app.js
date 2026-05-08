/* ════════════════════════════════════════════════════════════════════
   PRISM STUDIO — APP.JS
   GSAP · ScrollTrigger · Lenis · SplitType · Magnetic Cursor
   ════════════════════════════════════════════════════════════════════ */

// ─────────────────────────────────────────────────────────────────────────────
// 1. LENIS — smooth scroll init, synced to GSAP ticker
// ─────────────────────────────────────────────────────────────────────────────
const lenis = new Lenis({
    duration: 1.4,
    easing: t => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    orientation: 'vertical',
    smoothWheel: true,
    wheelMultiplier: 0.9,
    touchMultiplier: 1.5,
});

gsap.registerPlugin(ScrollTrigger);
lenis.on('scroll', ScrollTrigger.update);
gsap.ticker.add(t => lenis.raf(t * 1000));
gsap.ticker.lagSmoothing(0);

// ─────────────────────────────────────────────────────────────────────────────
// 2. CUSTOM CURSOR — dot + trailing ring + magnetic elements
// ─────────────────────────────────────────────────────────────────────────────
const cursorDot = document.getElementById('cursor-dot');
const cursorRing = document.getElementById('cursor-ring');
let mouse = { x: 0, y: 0 };
let ring = { x: 0, y: 0 };

window.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    gsap.set(cursorDot, { x: mouse.x, y: mouse.y });
});

gsap.ticker.add(() => {
    const lerp = (a, b, n) => a + (b - a) * n;
    ring.x = lerp(ring.x, mouse.x, 0.12);
    ring.y = lerp(ring.y, mouse.y, 0.12);
    gsap.set(cursorRing, { x: ring.x, y: ring.y });
});

// Hover states
document.querySelectorAll('button, a, select').forEach(el => {
    el.addEventListener('mouseenter', () => cursorRing.classList.add('hovered'));
    el.addEventListener('mouseleave', () => cursorRing.classList.remove('hovered'));
});
document.querySelectorAll('textarea').forEach(el => {
    el.addEventListener('mouseenter', () => cursorRing.classList.add('text-hovered'));
    el.addEventListener('mouseleave', () => cursorRing.classList.remove('text-hovered'));
});

// Magnetic physics
function applyMagnetic(selector, strength = 0.35, restEase = 0.7) {
    document.querySelectorAll(selector).forEach(el => {
        el.addEventListener('mousemove', e => {
            const r = el.getBoundingClientRect();
            const dx = e.clientX - (r.left + r.width / 2);
            const dy = e.clientY - (r.top + r.height / 2);
            gsap.to(el, { x: dx * strength, y: dy * strength, duration: 0.4, ease: 'power2.out' });
        });
        el.addEventListener('mouseleave', () => {
            gsap.to(el, { x: 0, y: 0, duration: restEase, ease: 'elastic.out(1, 0.4)' });
        });
    });
}

applyMagnetic('.magnetic', 0.35, 0.75);
applyMagnetic('.magnetic-sm', 0.2, 0.6);

// ─────────────────────────────────────────────────────────────────────────────
// 3. THEME TOGGLE
// ─────────────────────────────────────────────────────────────────────────────
const toggle = document.getElementById('theme-toggle');
let currentTheme = 'dark';

if (toggle) {
    toggle.addEventListener('click', e => {
        const opt = e.target.closest('.toggle-option');
        if (!opt) return;
        const val = opt.dataset.val;
        currentTheme = val;
        document.documentElement.setAttribute('data-theme', val);
        toggle.setAttribute('data-active', val);
        toggle.querySelectorAll('.toggle-option').forEach(o => o.classList.toggle('active', o.dataset.val === val));
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. HERO TEXT REVEAL — Line by line with SplitType
// ─────────────────────────────────────────────────────────────────────────────
// The hero lines are already wrapped in .line-mask / .line-inner in HTML.
const lineInners = document.querySelectorAll('#hero-title .line-inner');

const tlHero = gsap.timeline({ delay: 0.3 });
tlHero
    .to(lineInners, {
        y: 0,
        duration: 1.1,
        stagger: 0.12,
        ease: 'expo.out',
    })
    .to('#hero-sub', {
        opacity: 1,
        y: 0,
        duration: 1,
        ease: 'expo.out',
    }, '-=0.6')
    .from('.badge, .scroll-indicator, #btn-scroll-down', {
        opacity: 0,
        y: 15,
        duration: 0.8,
        stagger: 0.1,
        ease: 'expo.out',
    }, '-=0.9');

gsap.set('#hero-sub', { y: 30 });

// ─────────────────────────────────────────────────────────────────────────────
// 5. MARQUEE — gsap infinite loop
// ─────────────────────────────────────────────────────────────────────────────
const marqueeTrack = document.getElementById('marquee-track');
if (marqueeTrack) {
    gsap.to(marqueeTrack, {
        x: '-=33.33%',
        ease: 'none',
        duration: 16,
        repeat: -1,
        modifiers: {
            x: gsap.utils.unitize(x => parseFloat(x) % (marqueeTrack.scrollWidth / 3)),
        },
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. HORIZONTAL SCROLL — GSAP ScrollTrigger pinned section
// ─────────────────────────────────────────────────────────────────────────────
const studioTrack = document.getElementById('studio-track');
const panels = gsap.utils.toArray('.studio-panel');

if (studioTrack && panels.length) {
    // Calculate total horizontal scroll distance
    const getTotalWidth = () => {
        return -(studioTrack.scrollWidth - window.innerWidth);
    };

    const horizontalTween = gsap.to(studioTrack, {
        x: getTotalWidth,
        ease: 'none',
    });

    ScrollTrigger.create({
        trigger: '#studio',
        start: 'top top',
        end: () => `+=${studioTrack.scrollWidth - window.innerWidth}`,
        pin: '#studio-pin-wrapper',
        animation: horizontalTween,
        scrub: 1.2,
        invalidateOnRefresh: true,
        onUpdate: self => {
            // Parallax the background text slightly
            const bgText = document.querySelector('.hero-bg-text');
            if (bgText) {
                gsap.set(bgText, { x: -self.progress * 80 });
            }
        }
    });

    // Panel title reveals as they scroll into horizontal view
    panels.forEach((panel, i) => {
        const lineInners = panel.querySelectorAll('.line-inner');
        gsap.set(lineInners, { y: '110%' });

        ScrollTrigger.create({
            trigger: panel,
            containerAnimation: horizontalTween,
            start: 'left 80%',
            onEnter: () => {
                gsap.to(lineInners, {
                    y: 0,
                    duration: 1,
                    stagger: 0.1,
                    ease: 'expo.out',
                });
            },
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. FOOTER TITLE REVEAL
// ─────────────────────────────────────────────────────────────────────────────
const footerTitle = document.getElementById('footer-title');
if (footerTitle) {
    const splitFooter = new SplitType(footerTitle, { types: 'chars' });
    gsap.set(splitFooter.chars, { yPercent: 120, opacity: 0 });
    ScrollTrigger.create({
        trigger: footerTitle,
        start: 'top 85%',
        onEnter: () => {
            gsap.to(splitFooter.chars, {
                yPercent: 0,
                opacity: 1,
                duration: 0.9,
                stagger: 0.018,
                ease: 'back.out(1.8)',
            });
        },
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. SCROLL BUTTON — smooth scroll to studio
// ─────────────────────────────────────────────────────────────────────────────
const btnScrollDown = document.getElementById('btn-scroll-down');
if (btnScrollDown) {
    btnScrollDown.addEventListener('click', () => {
        lenis.scrollTo('#studio', { duration: 1.8, easing: t => Math.min(1, 1.001 - Math.pow(2, -10 * t)) });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. TOKEN COUNTER & ENGINE SELECTOR
// ─────────────────────────────────────────────────────────────────────────────
const sourceInput = document.getElementById('source-input');
const tokenCounter = document.getElementById('token-counter');
const engineSelect = document.getElementById('engine-select');
const presetRow = document.getElementById('preset-row');

const LIMITS = { bart: Infinity, gemini: 1_000_000 };
let MAX_TOKENS = LIMITS.bart;

function formatLimit(n) {
    if (n === Infinity) return '∞';
    return n >= 1000 ? n.toLocaleString() : n.toString();
}

function updateCounter() {
    const text = sourceInput ? sourceInput.value.trim() : '';
    const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
    const est = Math.floor(words * 1.3);
    if (tokenCounter) {
        const isUnlimited = MAX_TOKENS === Infinity;
        if (isUnlimited) {
            // BART chunked — show count, no warning possible
            const chunks = Math.max(1, Math.ceil(est / 900));
            tokenCounter.textContent = est === 0
                ? '000 TOKENS'
                : chunks > 1
                    ? `${est.toLocaleString()} TOKENS · ${chunks} CHUNKS`
                    : `${String(est).padStart(3, '0')} TOKENS`;
            tokenCounter.classList.remove('warn');
        } else {
            tokenCounter.textContent = est > MAX_TOKENS
                ? `⚠ ${est.toLocaleString()} / ${formatLimit(MAX_TOKENS)} — WILL TRUNCATE`
                : `${String(est).padStart(3, '0')} / ${formatLimit(MAX_TOKENS)} TOKENS`;
            tokenCounter.classList.toggle('warn', est > MAX_TOKENS);
        }
    }
}

if (engineSelect) {
    engineSelect.addEventListener('change', () => {
        const isGemini = engineSelect.value === 'gemini';
        MAX_TOKENS = isGemini ? LIMITS.gemini : LIMITS.bart;
        if (presetRow) presetRow.style.display = isGemini ? 'none' : '';
        updateCounter();
    });
}

if (sourceInput) sourceInput.addEventListener('input', updateCounter);

// ─────────────────────────────────────────────────────────────────────────────
// 10. SUMMARIZATION API CALL
// ─────────────────────────────────────────────────────────────────────────────
const outputDisplay = document.getElementById('output-display');
const btnSummarize = document.getElementById('btn-summarize');
const btnClear = document.getElementById('btn-clear');
const btnPolish = document.getElementById('btn-polish');
const loadingOverlay = document.getElementById('loading-overlay');
const presetSelect = document.getElementById('preset-select');

// Polish toggle state
let polishEnabled = false;
if (btnPolish) {
    btnPolish.addEventListener('click', () => {
        polishEnabled = !polishEnabled;
        btnPolish.textContent = polishEnabled ? '✦ GEMINI POLISH ON' : '✦ GEMINI POLISH OFF';
        btnPolish.style.borderColor = polishEnabled ? 'var(--lime)' : '';
        btnPolish.style.color = polishEnabled ? 'var(--lime)' : '';
    });
}

const PRESETS = {
    quick: { max_new_tokens: 64,  min_new_tokens: 15, num_beams: 4, length_penalty: 0.6 },
    notes: { max_new_tokens: 100, min_new_tokens: 25, num_beams: 5, length_penalty: 0.8 },
    deep:  { max_new_tokens: 160, min_new_tokens: 40, num_beams: 6, length_penalty: 1.0 },
};

if (btnClear) {
    btnClear.addEventListener('click', () => {
        if (sourceInput) sourceInput.value = '';
        if (outputDisplay) outputDisplay.value = '';
        if (tokenCounter) { tokenCounter.textContent = '000 TOKENS'; tokenCounter.classList.remove('warn'); }
    });
}

if (btnSummarize) {
    btnSummarize.addEventListener('click', async () => {
        const text = sourceInput ? sourceInput.value.trim() : '';
        if (!text) { alert('PLEASE PROVIDE A SOURCE DOCUMENT.'); return; }

        const engine = engineSelect ? engineSelect.value : 'bart';
        const preset = PRESETS[presetSelect ? presetSelect.value : 'notes'];
        if (loadingOverlay) loadingOverlay.classList.add('active');

        try {
        const body = engine === 'gemini'
                ? { text, engine: 'gemini', gemini_model: 'gemini-3.0-flash' }
                : {
                    text,
                    engine: 'bart',
                    max_new_tokens: preset.max_new_tokens,
                    min_new_tokens: preset.min_new_tokens,
                    num_beams: preset.num_beams,
                    length_penalty: preset.length_penalty,
                    polish: polishEnabled,
                    gemini_model: 'gemini-3.0-flash',
                };

            const res = await fetch('/api/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'API error');
            }

            const data = await res.json();
            if (outputDisplay) outputDisplay.value = data.summary;
            
            if (data.engine_used && data.engine_used.includes('polish failed')) {
                alert('Gemini Polish failed (likely due to API rate limits or quota). Displaying raw BART summary instead.');
            }

        } catch (err) {
            console.error(err);
            alert('INFERENCE FAILURE: ' + err.message);
        } finally {
            if (loadingOverlay) loadingOverlay.classList.remove('active');
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. NAV HIDE ON SCROLL
// ─────────────────────────────────────────────────────────────────────────────
const nav = document.getElementById('nav');
let lastY = 0;
lenis.on('scroll', ({ scroll }) => {
    if (!nav) return;
    nav.style.transform = scroll > lastY && scroll > 80
        ? 'translateY(-100%)'
        : 'translateY(0)';
    lastY = scroll;
    nav.style.transition = 'transform 0.5s cubic-bezier(0.19,1,0.22,1)';
});

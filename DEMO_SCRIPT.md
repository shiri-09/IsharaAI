# Ishara AI: 3 minute demo video script

Total: 3:00. Screen recording with voice over. Two people help: one signs (ISL), one uses the eyes (AAC). The voice over is in plain words; read it at a calm pace.

## Before you record (10 minutes, once)

- Backend on port 8000, Ollama running with the Qwen model loaded (send one test message first so the model is warm).
- Sign in once and calibrate the eyes once on the day, at the screen size you will record. Calibrate again if you resize the window.
- Chrome at 100% zoom, window maximised, notifications off. Close the "--no-sandbox" bar in the agent's Chrome window.
- Sign in as a demo account. Run "Order lunch from the campus canteen" once beforehand so your name, phone and room are saved (the agent will then fill them in front of the camera). To show the memory moment, delete just the "address" memory in the Memory list, so one question is still asked on camera.
- Good light on faces. Sit a hand to an arm's length from the camera.
- Do each live moment once as a rehearsal. Keep the best take.

Sentences the sign reader knows (use only these on camera): "I need water", "Hi how are you", "Had your food", "I am very happy", "Congratulations".
Rehearse "I need water" and "Hi how are you" most. If one is misread on camera, use the "Not right? I signed:" buttons under the recognised sentence and say "it learns from this".

## The script

| Time | On screen | Voice over | Action |
|---|---|---|---|
| 0:00 to 0:12 | Black screen, the Ishara AI intro (first lines only: "Hello." then "It takes one second to say. For some people, it takes an interpreter.") | "Saying hello takes one second. For Deaf people, and for people who cannot speak or move their hands, it takes an interpreter. Ishara AI removes that wait." | Let the intro play about 12 seconds, then press Esc. |
| 0:12 to 0:22 | Login page | "Ishara AI is one app for two kinds of people. You sign in once, and your choices follow you." | Type the demo email and password, press Sign in. |
| 0:22 to 0:35 | Setup page: two big cards | "First, you choose how you talk. Sign language on the left, or eyes and face on the right. You can pick with your eyes." | Hover the cursor over the ISL card, click it. (The AAC card is shown later.) |
| 0:35 to 1:10 | **ISL: AI conversation** | "In sign mode the camera reads whole sentences, not just letters. Watch." | Start camera. Sign "I need water", lower hands. The recognised sentence appears with its percentage and the AI replies. Point to the reply, then click the Hindi button under the reply: it shows the translation and speaks it. |
| | | "It answers in text and speech in Hindi, Kannada, Tamil, Telugu and English, and everything is saved in History." | Briefly open History. |
| 1:10 to 1:25 | **Reply in sign** | "The talk works both ways. A hearing person types or speaks, and an avatar signs it back." | Type "Hi how are you", press "Sign it". The avatar plays. Then type a name, for example "Asha", and press the spell button next to it (finger-spell every word). |
| 1:25 to 1:35 | **Fingerspelling** | "Names are spelled letter by letter. All 26 ISL letters are built in." | Show A to Z in Sign library, or spell one letter on camera. |
| 1:35 to 2:15 | **ISL: Agentic orchestration** (Live browser tab) | "Now the agent. It does real tasks in a real browser. I will ask it to order lunch." | Type "Order lunch from the campus canteen", Run. Chrome opens. Narrate while it works: |
| | The question: "Which one would you like?" | "It asks me, because the choice is mine." | Click a dish. |
| | The memory | "It knows my name and phone from last time and fills them in. It asked for my room once and will remember it." | Answer the room question. |
| | "Shall I click Place order?" | "Before it spends or sends anything, it asks me. Passwords and card numbers it never touches." | Click "Yes, go ahead". |
| | Order number, "Checked on the page" | "And it checks the page for proof: here is the order number." | Show the plan ticked off and the order number. |
| 2:15 to 2:25 | Change mode, pick AAC | "Now the second half: someone who can only use their eyes." | Click Change mode in the sidebar, then choose the eyes and face card. |
| 2:25 to 2:55 | **AAC: AI conversation** (eye typing) | "No mouse, no keyboard. The silver circle shows where I look. Eyebrows up means yes, mouth open means no." | Look at the keyboard letters to type "water" (use word suggestions), eyebrows up to choose a key or press Speak. The reply is spoken. |
| 2:55 to 3:00 | **AAC: Agentic orchestration** then logo | "Even the agent works with eyes: the questions become big tiles, and my eyebrows answer them." | Look at a dish tile, eyebrows up; then look at Yes and raise eyebrows. Cut to the logo. |
| 3:00 | End card | "Ishara AI. No one left out of the conversation." | Fade to the logo and the GitHub address. |

(Total: aim for 2:55 and let the end card be the last 5 seconds.)

## Short voice over, clean read (about 380 words)

Saying hello takes one second. For Deaf people, and for people who cannot speak or move their hands, it takes an interpreter. Ishara AI removes that wait.

It is one app for two kinds of people. You sign in once, then choose how you talk: with sign language, or with your eyes and face.

In sign mode the camera reads whole sentences, not just letters. I sign "I need water", and the reader shows what it understood and how sure it is. The AI answers in text and speech, in Hindi, Kannada, Tamil, Telugu and English, and everything is saved in History.

The talk works both ways. A hearing person types or speaks, and an avatar signs it back. Names are spelled letter by letter, with all 26 ISL letters built in.

Now the agent. It does real tasks in a real browser. I ask it to order lunch. It plans the steps and asks me what only I can decide: which dish. It remembers my name and phone from last time, and asks for my room once. Before it spends or sends anything, it asks me. Passwords and card numbers it never touches. And it checks the page for proof: here is the order number.

The second half is for someone who can only use their eyes. There is no mouse and no keyboard. The silver circle shows where I look. Eyebrows up means yes, mouth open means no. I type with my eyes on an on-screen keyboard, and the app speaks for me. Even the agent works with eyes: its questions become big tiles, and my eyebrows answer them.

Ishara AI. No one left out of the conversation.

## What to say if you are asked

- **How accurate is it?** On the 20 core sentences, 86.3% right first time and 95.7% in the top three, tested by cross-validation on recordings the model never learned from. Those 20 were picked after earlier tests, so it is slightly optimistic. On all 99 sentences it is 56.9%.
- **Is it a closed list?** Yes, 20 set sentences. Anything else is fingerspelled or typed. New recordings, including your own, are learned in about 15 seconds with the fine-tune button.
- **Who trained it?** We trained it on our own computer, with a PyTorch Transformer on the public ISL-CSLTR videos (CC BY 4.0) plus our own recordings. It runs locally; no cloud AI trained it.
- **Is the agent safe?** It asks before paying or sending, never types passwords, cards or OTPs, and only remembers name, phone, email and address. You can delete any memory.
- **Eye tracking accuracy?** It shows the measured error in pixels after a 30 second setup. It is tuned per person and per screen, so we calibrate on the day.

## Safe fallbacks if something fails on camera

- A sign is misread: use "Not right? I signed:" to fix it and say "it learns from this". Or type the sentence in the box.
- The eye circle drifts: look at the panel and press "Re-center", or click with the mouse and keep talking.
- The agent is slow: it is the local model thinking; keep narrating the plan on the screen. Do not click while it works.
- Ollama is cold: send one message before recording.

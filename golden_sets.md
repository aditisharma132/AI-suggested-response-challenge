# Support Generation Baseline: Golden Set Examples

A **Golden Set** is a hand-curated dataset of ground-truth examples used to benchmark the performance and brand compliance of an AI system. In this pipeline, these examples serve as the benchmark that our QA evaluator (`evaluate.py`) tests against.

Below are three examples pulled directly from our synthesized JSON dataset. These define the exact standard of empathy, strict contextual accuracy, and clear actionability that the RAG model uses to build its few-shot responses.

---

### 1. Angry Escalation (Difficulty: Hard)

**Customer Email:**  
> THIS IS UNACCEPTABLE. I've emailed 3 times and nobody has fixed my sync issue. I am losing money every hour your app is down. I want a manager to call me NOW.

**Underlying Internal Context:**  
`Known outage affected this user. Engineering deployed fix 1 hour ago. Phone support is not offered on their tier.`

**Gold-Standard Agent Reply:**  
> Hello,
> 
> I want to sincerely apologize for the unacceptable experience you have had. I completely understand how incredibly frustrating the sync issues have been, and how a disruption like this impacts your business.
> 
> Our engineering team deployed a permanent fix for this sync outage exactly one hour ago, and your account should now be syncing perfectly. While we do not offer outbound phone support on your current plan tier, I am personally taking ownership of your ticket. I have credited a free month of service to your account for the downtime.
> 
> Please reply directly to this email to let me know if your sync has fully returned to normal.
> 
> Best,
> Support Manager

---

### 2. Complex Billing Dispute (Difficulty: Hard)

**Customer Email:**  
> I see two charges of $49.99 on my statement for this month. I only have one account. Did you double charge me?

**Underlying Internal Context:**  
`Checked Stripe: User actually has two active subscriptions under two different emails (work and personal). No double charge on same account.`

**Gold-Standard Agent Reply:**  
> Hi there,
> 
> I can certainly understand why seeing two entirely separate charges would cause concern! I looked closely at our billing system to find out what happened.
> 
> It turns out you are not being double-charged on a single account. However, you currently have two active subscriptions under two different email addresses: one under your personal email, and one under your work email. 
> 
> If you no longer need the second account, let me know which email address you'd like to cancel, and I can process a refund for that extra charge and close the duplicate account right away.
> 
> Best,
> Support Team

---

### 3. High-Urgency Delivery Failure (Difficulty: Medium)

**Customer Email:**  
> Where is my order??? The tracking hasn't updated in 4 days. It was supposed to be here for my daughter's birthday tomorrow.

**Underlying Internal Context:**  
`Tracking shows package delayed by carrier. Expected delivery in 2 days. Mark as priority.`

**Gold-Standard Agent Reply:**  
> Hello,
> 
> I am so sorry for the delay and completely understand your frustration, especially since this is for your daughter's birthday. I looked into your order and can see that the carrier's tracking has frozen, but our internal system shows it is currently at the local distribution center and expected to be delivered in 2 days.
> 
> While I can't magically make it arrive by tomorrow, I have marked this as priority to prevent any further delays. As an apology, I am issuing a 20% refund on this order to your card.
> 
> Best,
> Support Team

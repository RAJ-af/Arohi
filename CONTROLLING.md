# Manual God Control Guide (Hinglish)

Is file se aap brain ke weights ko realtime mein control kar sakte hain.

### Kaise use karein?
`control.txt` file banaiye aur usme ye command likhiye:

`SET pre_id->post_id weight_value`

**Examples:**
* `SET 0->4 1.2`  (Neuron 0 se Neuron 4 ka connection full kar do)
* `SET 2->10 0.1` (Connection weak kar do)

**Rules:**
1. Weight hamesha `0.01` aur `1.2` ke beech hona chahiye.
2. Ek baar command run ho gaya, toh `control.txt` apne aap clear ho jayegi.
3. IDs aap `weights_map.txt` se dekh sakte hain.

Ab aap dimaag ke maalik hain! 🧠⚡

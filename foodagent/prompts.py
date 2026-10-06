"""Prompt templates for the reasoning agent, the experts, the response generator and the baselines."""

ELEMENT_ANALYSIS = """You are a nutrition reasoning agent. Analyze this photo of a dish.
{target_clause}
Decompose the dish into its individual food elements: every distinct food or ingredient component that is visible, including sauces, dressings, oils, garnishes and condiments.
For each element output:
- "name": short common food name (e.g. "scrambled eggs", "bacon", "white rice")
- "description": appearance, cooking method, rough size and position on the plate
- "bbox_2d": [x1, y1, x2, y2] bounding box of the element in the image, in pixels
- "importance": "main" if the element contributes substantially to the calories or macronutrients of the dish; "minor" for herbs, garnishes, small condiments, or low-calorie trace items
- "visually_clear": true only if the element is a distinct, well-separated solid item with a regular shape whose volume can be measured from the image (e.g. whole pieces of meat, bread, fruit pieces, a potato); false if it is mixed, scattered, amorphous, flattened, occluded, or blended with other items (e.g. scrambled eggs, mixed pasta, dressed salad, stew, rice covered by sauce)
- "density_g_per_cm3": typical density of the element as served
- "estimated_mass_g": your rough estimate of the mass of this element in the image
- "per_100g": {{"calories": kcal, "fat": g, "carb": g, "protein": g}} typical nutrition per 100 g of the element as served
Respond with JSON only, in the form {{"elements": [ ... ]}}."""

TARGET_CLAUSE = (
    "The user specifically asks about these food items: {names}. Each of them MUST appear as an element, "
    "using exactly that name."
)

LLM_DIRECT = """You are a nutrition expert. The following minor or easy-to-estimate food elements were identified in this dish:
{element_list}
Using your nutrition knowledge and the image, estimate for each element the mass present in the image and its total nutrition.
Respond with JSON only, in the form {{"elements": [{{"index": <element number>, "name": "...", "mass_g": number, "calories": kcal, "fat": g, "carb": g, "protein": g}}]}}."""

RAG_UNITS = """You are a nutrition expert. The first image is a photo of a dish. The following food elements in the dish are nutritionally important but hard to measure visually. For each element, reference servings retrieved from the NutritionX database are listed with their unit size and per-unit nutrition{photo_note}.
{element_blocks}
For each element, choose the reference that best matches what is in the dish photo, then estimate how many of its units are present in the dish (decimals allowed, e.g. 2.5). Compare the visible amount with the reference unit size.
Respond with JSON only, in the form {{"elements": [{{"element": <element number>, "reference": <reference number>, "num_units": number, "reasoning": "short"}}]}}."""

DIRECT_TOTAL = """Estimate the nutrition of the entire dish in this image.
Respond with JSON only, in the form {"calories": kcal, "mass": grams, "fat": grams, "carb": grams, "protein": grams}."""

DIRECT_REFER = """Estimate the nutrition of the following food items in the dish in this image: {names}.
Give the values for each item separately (only the amount of that item present in the dish).
Respond with JSON only, in the form {{"items": [{{"name": "...", "calories": kcal, "mass": grams, "fat": grams, "carb": grams, "protein": grams}}]}}."""

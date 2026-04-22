/**
 * HiDDeN Cryptography Client
 */
const bufToHex = (buffer) => Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, '0')).join('');
const str2ab = (str) => {
    const binaryString = window.atob(str);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
    return bytes.buffer;
};

// HELPER: Convert Raw WebCrypto signature to DER
function derEncodeSignature(rawSignature) {
    const bytes = new Uint8Array(rawSignature);
    const r = bytes.slice(0, bytes.length / 2);
    const s = bytes.slice(bytes.length / 2);
    const toDer = (part) => {
        let hex = Array.from(part).map(b => b.toString(16).padStart(2, '0')).join('');
        if (parseInt(hex[0], 16) >= 8) hex = '00' + hex;
        const len = (hex.length / 2).toString(16).padStart(2, '0');
        return `02${len}${hex}`;
    };
    const rDer = toDer(r); const sDer = toDer(s);
    const totalLen = (rDer.length + sDer.length) / 2;
    return `30${totalLen.toString(16).padStart(2, '0')}${rDer}${sDer}`;
}

export async function generateKeyPair() {
    const keyPair = await window.crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, true, ["sign", "verify"]);
    const privExport = await window.crypto.subtle.exportKey("pkcs8", keyPair.privateKey);
    const privBase64 = btoa(String.fromCharCode(...new Uint8Array(privExport)));
    const privatePem = `-----BEGIN PRIVATE KEY-----\n${privBase64.match(/.{1,64}/g).join('\n')}\n-----END PRIVATE KEY-----`;
    const pubExport = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
    return { privatePem, publicHex: bufToHex(pubExport) };
}

export async function importPrivateKey(pem) {
    const b64 = pem.replace(/-----(BEGIN|END) PRIVATE KEY-----/g, '').replace(/\s/g, '');
    return await window.crypto.subtle.importKey("pkcs8", str2ab(b64), { name: "ECDSA", namedCurve: "P-256" }, true, ["sign"]);
}

export async function getPublicKeyHexFromPEM(pem) {
    const privKey = await importPrivateKey(pem);
    const jwk = await window.crypto.subtle.exportKey("jwk", privKey);
    const pubKey = await window.crypto.subtle.importKey("jwk", { kty: "EC", crv: "P-256", x: jwk.x, y: jwk.y, ext: true }, { name: "ECDSA", namedCurve: "P-256" }, true, ["verify"]);
    const spki = await window.crypto.subtle.exportKey("spki", pubKey);
    return bufToHex(spki);
}

export async function signMetadata(privateKey, metadataObj) {
    const jsonString = JSON.stringify(metadataObj);
    const data = new TextEncoder().encode(jsonString);
    const signature = await window.crypto.subtle.sign({ name: "ECDSA", hash: { name: "SHA-256" } }, privateKey, data);
    return { signatureHex: derEncodeSignature(signature), jsonString: jsonString };
}